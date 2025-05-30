#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <chrono>
#include <stdexcept>
#include <iomanip>
#include <omp.h>

using namespace std;
using namespace std::chrono;

class Matrix {
private:
    vector<vector<double>> data;
    int rows;
    int cols;

public:
    Matrix() : rows(0), cols(0) {}
    
    Matrix(int r, int c) : rows(r), cols(c), data(r, vector<double>(c)) {}
    
    void readFromFile(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            throw runtime_error("Не удалось открыть файл: " + filename);
        }

        string line;
        while (getline(file, line)) {
            if (line.empty()) continue;

            vector<double> row;
            stringstream ss(line);
            double val;
            while (ss >> val) {
                row.push_back(val);
            }

            if (cols == 0) {
                cols = row.size();
            } else if (row.size() != cols) {
                throw runtime_error("Несогласованное количество столбцов в файле " + filename);
            }
            data.push_back(row);
        }
        rows = data.size();
    }

    void writeToFile(const string& filename, long long duration_ms = -1) const {
        ofstream file(filename);
        if (!file.is_open()) {
            throw runtime_error("Не удалось открыть файл: " + filename);
        }

        if (duration_ms >= 0) {
            file << "Время выполнения: " << duration_ms << " мс\n";
            file << "Размер матрицы: " << rows << "x" << cols << "\n\n";
        }

        for (const auto& row : data) {
            for (double val : row) {
                file << val << " ";
            }
            file << "\n";
        }
    }

    Matrix multiply(const Matrix& other) const {
        if (cols != other.rows) {
            throw runtime_error("Несовместимые размеры матриц для умножения");
        }

        Matrix result(rows, other.cols);
        const int block_size = 32; // Оптимальный размер блока для Apple M1/M2
        
        // Основное распараллеливание с блочной оптимизацией
        #pragma omp parallel for collapse(2) schedule(guided)
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < other.cols; ++j) {
                double sum = 0.0;
                
                // Векторизация внутреннего цикла
                #pragma omp simd reduction(+:sum)
                for (int k = 0; k < cols; ++k) {
                    sum += data[i][k] * other.data[k][j];
                }
                
                result.data[i][j] = sum;
            }
        }

        return result;
    }

    int getRows() const { return rows; }
    int getCols() const { return cols; }
};

int main() {
    try {
        // Настройка OpenMP для macOS
        omp_set_num_threads(omp_get_max_threads());
        cout << "Используется потоков: " << omp_get_max_threads() << endl;
        cout << "Процессорные ядер: " << omp_get_num_procs() << endl;

        Matrix A, B;
        
        cout << "Чтение матрицы A..." << endl;
        auto start_read = high_resolution_clock::now();
        A.readFromFile("matrix_a.txt");
        auto end_read = high_resolution_clock::now();
        cout << "Матрица A: " << A.getRows() << "x" << A.getCols() << endl;
        cout << "Время чтения A: " << duration_cast<milliseconds>(end_read - start_read).count() << " мс" << endl;

        cout << "Чтение матрицы B..." << endl;
        start_read = high_resolution_clock::now();
        B.readFromFile("matrix_b.txt");
        end_read = high_resolution_clock::now();
        cout << "Матрица B: " << B.getRows() << "x" << B.getCols() << endl;
        cout << "Время чтения B: " << duration_cast<milliseconds>(end_read - start_read).count() << " мс" << endl;

        if (A.getCols() != B.getRows()) {
            throw runtime_error("Размеры матриц не подходят для умножения");
        }

        cout << "Умножение матриц..." << endl;
        auto start = high_resolution_clock::now();
        Matrix C = A.multiply(B);
        auto end = high_resolution_clock::now();

        auto duration = duration_cast<milliseconds>(end - start);
        cout << "Время умножения: " << duration.count() << " мс" << endl;

        cout << "Запись результата в result.txt..." << endl;
        start = high_resolution_clock::now();
        C.writeToFile("result.txt", duration.count());
        end = high_resolution_clock::now();
        cout << "Время записи: " << duration_cast<milliseconds>(end - start).count() << " мс" << endl;

        cout << "Готово!" << endl;

    } catch (const exception& e) {
        cerr << "Ошибка: " << e.what() << endl;
        return 1;
    }

    return 0;
}
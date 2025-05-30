#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <chrono>
#include <stdexcept>
#include <iomanip>
#include <omp.h> // Добавляем заголовочный файл OpenMP

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

        // Добавляем информацию о времени выполнения в начало файла
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

        // Распараллеливаем внешний цикл с помощью OpenMP
        #pragma omp parallel for
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < other.cols; ++j) {
                double sum = 0.0;
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
        Matrix A, B;
        
        cout << "Чтение матрицы A..." << endl;
        A.readFromFile("matrix_a.txt");
        cout << "Матрица A: " << A.getRows() << "x" << A.getCols() << endl;

        cout << "Чтение матрицы B..." << endl;
        B.readFromFile("matrix_b.txt");
        cout << "Матрица B: " << B.getRows() << "x" << B.getCols() << endl;

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
        C.writeToFile("result.txt", duration.count());
        cout << "Готово!" << endl;

    } catch (const exception& e) {
        cerr << "Ошибка: " << e.what() << endl;
        return 1;
    }

    return 0;
}
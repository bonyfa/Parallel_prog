#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <chrono>
#include <stdexcept>

using namespace std;
using namespace std::chrono;

class Matrix {
private:
    vector<vector<double>> data;
    int rows;
    int cols;

public:
    // Конструктор по умолчанию
    Matrix() : rows(0), cols(0) {}
    
    // Конструктор с размерами
    Matrix(int r, int c) : rows(r), cols(c), data(r, vector<double>(c)) {}
    
    // Чтение матрицы из файла
    void readFromFile(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            throw runtime_error("Cannot open file: " + filename);
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
                throw runtime_error("Inconsistent number of columns in " + filename);
            }
            data.push_back(row);
        }
        rows = data.size();
    }

    // Запись матрицы в файл
    void writeToFile(const string& filename) const {
        ofstream file(filename);
        if (!file.is_open()) {
            throw runtime_error("Cannot open file: " + filename);
        }

        for (const auto& row : data) {
            for (double val : row) {
                file << val << " ";
            }
            file << "\n";
        }
    }

    // Умножение матриц с транспонированием
    Matrix multiplyWithTranspose(const Matrix& other) const {
        if (cols != other.rows) {
            throw runtime_error("Matrix dimensions mismatch for multiplication");
        }

        Matrix result(rows, other.cols);
        Matrix otherTransposed = other.transpose();

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < other.cols; ++j) {
                double sum = 0.0;
                for (int k = 0; k < cols; ++k) {
                    sum += data[i][k] * otherTransposed.data[j][k];
                }
                result.data[i][j] = sum;
            }
        }

        return result;
    }

    // Транспонирование матрицы
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result.data[j][i] = data[i][j];
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
        
        cout << "Reading matrix A..." << endl;
        A.readFromFile("matrix_a.txt");
        cout << "Matrix A: " << A.getRows() << "x" << A.getCols() << endl;

        cout << "Reading matrix B..." << endl;
        B.readFromFile("matrix_b.txt");
        cout << "Matrix B: " << B.getRows() << "x" << B.getCols() << endl;

        if (A.getCols() != B.getRows()) {
            throw runtime_error("Matrix dimensions are not compatible for multiplication");
        }

        cout << "Multiplying matrices..." << endl;
        auto start = high_resolution_clock::now();
        Matrix C = A.multiplyWithTranspose(B);
        auto end = high_resolution_clock::now();

        auto duration = duration_cast<milliseconds>(end - start);
        cout << "Multiplication time: " << duration.count() << " ms" << endl;

        cout << "Writing result to result.txt..." << endl;
        C.writeToFile("result.txt");
        cout << "Done!" << endl;

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}
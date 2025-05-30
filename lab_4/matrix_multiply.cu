#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <chrono>
#include <stdexcept>
#include <iomanip>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

// CUDA kernel для умножения матриц
__global__ void matrixMultiplyKernel(const double* A, const double* B, double* C, 
                                    int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        double sum = 0.0;
        for (int k = 0; k < colsA; ++k) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

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

        // Выводим только первые 10x10 элементов для больших матриц
        int outputRows = min(10, rows);
        int outputCols = min(10, cols);

        for (int i = 0; i < outputRows; i++) {
            for (int j = 0; j < outputCols; j++) {
                file << fixed << setprecision(6) << data[i][j] << " ";
            }
            file << "\n";
        }
    }

    Matrix multiply(const Matrix& other) const {
        if (cols != other.rows) {
            throw runtime_error("Несовместимые размеры матриц для умножения");
        }

        Matrix result(rows, other.cols);

        // Подготовка данных для CUDA
        double *d_A, *d_B, *d_C;
        size_t sizeA = rows * cols * sizeof(double);
        size_t sizeB = other.rows * other.cols * sizeof(double);
        size_t sizeC = rows * other.cols * sizeof(double);

        // Выделение памяти на GPU
        cudaMalloc(&d_A, sizeA);
        cudaMalloc(&d_B, sizeB);
        cudaMalloc(&d_C, sizeC);

        // Преобразование данных в одномерные массивы
        vector<double> flatA(rows * cols);
        vector<double> flatB(other.rows * other.cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                flatA[i * cols + j] = data[i][j];
            }
        }
        for (int i = 0; i < other.rows; ++i) {
            for (int j = 0; j < other.cols; ++j) {
                flatB[i * other.cols + j] = other.data[i][j];
            }
        }

        // Копирование данных на GPU
        cudaMemcpy(d_A, flatA.data(), sizeA, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, flatB.data(), sizeB, cudaMemcpyHostToDevice);

        // Настройка размеров блоков и сетки
        dim3 blockDim(16, 16);
        dim3 gridDim((other.cols + blockDim.x - 1) / blockDim.x,
                    (rows + blockDim.y - 1) / blockDim.y);

        // Запуск CUDA kernel
        matrixMultiplyKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols, other.cols);

        // Копирование результата обратно на CPU
        vector<double> flatC(rows * other.cols);
        cudaMemcpy(flatC.data(), d_C, sizeC, cudaMemcpyDeviceToHost);

        // Преобразование результата обратно в двумерный массив
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < other.cols; ++j) {
                result.data[i][j] = flatC[i * other.cols + j];
            }
        }

        // Освобождение памяти GPU
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

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

        cout << "Умножение матриц с использованием CUDA..." << endl;
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
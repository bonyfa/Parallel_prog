#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <chrono>
#include <stdexcept>
#include <iomanip>
#include <mpi.h>

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
    
    void readFromFile(const string& filename, int rank = 0) {
        ifstream file;
        if (rank == 0) {
            file.open(filename);
            if (!file.is_open()) {
                throw runtime_error("Не удалось открыть файл: " + filename);
            }
        }

        string line;
        if (rank == 0) {
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

        // Рассылаем размеры матрицы всем процессам
        MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank != 0) {
            data.resize(rows, vector<double>(cols));
        }

        // Рассылаем данные матрицы всем процессам
        for (int i = 0; i < rows; ++i) {
            MPI_Bcast(data[i].data(), cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }

    void writeToFile(const string& filename, long long duration_ms = -1, int rank = 0) const {
        if (rank != 0) return;

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

    Matrix multiply(const Matrix& other, int rank, int size) const {
        if (cols != other.rows) {
            throw runtime_error("Несовместимые размеры матриц для умножения");
        }

        Matrix result(rows, other.cols);
        const int block_size = 32;
        
        // Распределяем строки матрицы между процессами
        int rows_per_process = rows / size;
        int remainder = rows % size;
        int start_row = rank * rows_per_process + min(rank, remainder);
        int end_row = start_row + rows_per_process + (rank < remainder ? 1 : 0);

        // Каждый процесс вычисляет свою часть матрицы
        for (int i = start_row; i < end_row; ++i) {
            for (int j = 0; j < other.cols; ++j) {
                double sum = 0.0;
                for (int k = 0; k < cols; ++k) {
                    sum += data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }

        // Собираем результаты на процессе 0
        if (rank == 0) {
            for (int src = 1; src < size; ++src) {
                int src_start = src * rows_per_process + min(src, remainder);
                int src_end = src_start + rows_per_process + (src < remainder ? 1 : 0);
                for (int i = src_start; i < src_end; ++i) {
                    MPI_Recv(result.data[i].data(), other.cols, MPI_DOUBLE, src, 0, 
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        } else {
            for (int i = start_row; i < end_row; ++i) {
                MPI_Send(result.data[i].data(), other.cols, MPI_DOUBLE, 0, 0, 
                        MPI_COMM_WORLD);
            }
        }

        return result;
    }

    int getRows() const { return rows; }
    int getCols() const { return cols; }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try {
        if (rank == 0) {
            cout << "Используется процессов: " << size << endl;
        }

        Matrix A, B;
        
        if (rank == 0) cout << "Чтение матрицы A..." << endl;
        auto start_read = high_resolution_clock::now();
        A.readFromFile("matrix_a.txt", rank);
        auto end_read = high_resolution_clock::now();
        if (rank == 0) {
            cout << "Матрица A: " << A.getRows() << "x" << A.getCols() << endl;
            cout << "Время чтения A: " << duration_cast<milliseconds>(end_read - start_read).count() << " мс" << endl;
        }

        if (rank == 0) cout << "Чтение матрицы B..." << endl;
        start_read = high_resolution_clock::now();
        B.readFromFile("matrix_b.txt", rank);
        end_read = high_resolution_clock::now();
        if (rank == 0) {
            cout << "Матрица B: " << B.getRows() << "x" << B.getCols() << endl;
            cout << "Время чтения B: " << duration_cast<milliseconds>(end_read - start_read).count() << " мс" << endl;
        }

        if (A.getCols() != B.getRows()) {
            if (rank == 0) {
                cerr << "Ошибка: Размеры матриц не подходят для умножения" << endl;
            }
            MPI_Finalize();
            return 1;
        }

        if (rank == 0) cout << "Умножение матриц..." << endl;
        auto start = high_resolution_clock::now();
        Matrix C = A.multiply(B, rank, size);
        auto end = high_resolution_clock::now();

        auto duration = duration_cast<milliseconds>(end - start);
        if (rank == 0) {
            cout << "Время умножения: " << duration.count() << " мс" << endl;
            cout << "Запись результата в result.txt..." << endl;
            start = high_resolution_clock::now();
            C.writeToFile("result.txt", duration.count(), rank);
            end = high_resolution_clock::now();
            cout << "Время записи: " << duration_cast<milliseconds>(end - start).count() << " мс" << endl;
            cout << "Готово!" << endl;
        }

    } catch (const exception& e) {
        if (rank == 0) {
            cerr << "Ошибка: " << e.what() << endl;
        }
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}
import numpy as np

def read_matrix(filename):
    """Читает матрицу из файла, автоматически определяя размеры"""
    with open(filename) as f:
        data = []
        for line in f:
            # Пропускаем пустые строки
            if not line.strip():
                continue
            # Добавляем строку матрицы
            row = list(map(float, line.split()))
            data.append(row)
        
        # Проверяем, что все строки имеют одинаковую длину
        if data:
            n_cols = len(data[0])
            for row in data:
                if len(row) != n_cols:
                    raise ValueError("Все строки матрицы должны иметь одинаковую длину")
        
        return np.array(data)

def compare_matrices(mat1, mat2, tol=1e-8):
    """Сравнивает две матрицы с заданной точностью"""
    if mat1.shape != mat2.shape:
        print(f"Размеры не совпадают: {mat1.shape} vs {mat2.shape}")
        return False
    
    if not np.allclose(mat1, mat2, atol=tol):
        print("Матрицы различаются!")
        diff = np.abs(mat1 - mat2)
        print("Максимальное различие:", np.max(diff))
        return False
    
    print("Матрицы совпадают!")
    return True

# Пример использования
try:
    A = read_matrix("/Users/srd/Desktop/parallel/Parallel_prog/lab_1/matrix_a.txt")
    B = read_matrix("/Users/srd/Desktop/parallel/Parallel_prog/lab_1/matrix_b.txt")
    C = read_matrix("/Users/srd/Desktop/parallel/Parallel_prog/lab_1/result.txt")
    
    print("Размер матрицы A:", A.shape)
    print("Размер матрицы B:", B.shape)
    print("Размер матрицы результата:", C.shape)
    
    # Проверяем возможность умножения
    if A.shape[1] != B.shape[0]:
        print("Ошибка: матрицы нельзя перемножить (A.columns != B.rows)")
    else:
        C_ref = A @ B
        compare_matrices(C, C_ref)
        
except FileNotFoundError as e:
    print(f"Ошибка: файл не найден - {e.filename}")
except ValueError as e:
    print(f"Ошибка в данных: {str(e)}")
except Exception as e:
    print(f"Произошла ошибка: {str(e)}")
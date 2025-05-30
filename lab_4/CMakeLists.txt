cmake_minimum_required(VERSION 3.8)
project(matrix_multiply CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(matrix_multiply matrix_multiply.cu)
set_target_properties(matrix_multiply PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON) 
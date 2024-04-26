cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(CudaExamples LANGUAGES C CUDA)

add_executable(dynamic_parallel dynamic_parallel.cu)
set_target_properties(dynamic_parallel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(static_variable static_variable.cu)
set_target_properties(static_variable PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(run_cuda_test test.cu)
set_target_properties(run_cuda_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
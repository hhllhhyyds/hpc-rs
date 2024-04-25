cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(RunCudaTest LANGUAGES C CUDA)

add_executable(run_cuda_test test.cu)
set_target_properties(run_cuda_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

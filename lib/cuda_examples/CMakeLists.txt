cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(CudaExamples LANGUAGES C CUDA)

macro(add_example example_name)
    add_executable(${example_name} ${example_name}.cu)
    set_target_properties(${example_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endmacro(add_example example_name)

add_example(dynamic_parallel)
add_example(static_variable)
add_example(run_cuda_test)
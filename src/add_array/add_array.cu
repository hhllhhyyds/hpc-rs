#include <cuda_runtime.h>

#include "../cuda_common/cuda_check.h"

__global__ void add_array_kernel(float *A, float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

extern "C" void add_array(float *A, float *B, float *C, int N, int grid, int block)
{

    float *d_a, *d_b, *d_c;

    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, B, N * sizeof(float), cudaMemcpyHostToDevice));

    add_array_kernel<<<grid, block>>>(d_a, d_b, d_c, N);

    CUDA_CHECK(cudaMemcpy(C, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}
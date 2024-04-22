#include <stdio.h>

extern "C" __global__ void cuda_hello(){
    int i = threadIdx.x;
    printf("Hello World from GPU, thread %d!\n", i);
}

extern "C" __global__ void check_index() {
    printf("threadIdx: (%d, %d, %d) blockIdx: (%d, %d, %d) blockDim: (%d, %d, %d) gridDim: (%d, %d, %d)\n",
        threadIdx.x, threadIdx.y, threadIdx.z, 
        blockIdx.x, blockIdx.y, blockIdx.z,
        blockDim.x, blockDim.y, blockDim.z,
        gridDim.x, gridDim.y, gridDim.z 
    );
}
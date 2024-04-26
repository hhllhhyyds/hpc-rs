#include <stdio.h>

extern "C" __global__ void nested_hello_world(int const iSize,int iDepth) {
    int tid = threadIdx.x;
    printf("Recursion=%d: Hello World from thread %d block %d\n", iDepth, tid, blockIdx.x);
    // condition to stop recursive execution
    if (iSize == 1) 
    {
        return;
    }
    // reduce block size to half
    int nthreads = iSize >> 1;
    // thread 0 launches child grid recursively
    if(tid == 0 && nthreads > 0) 
    {
        nested_hello_world<<<1, nthreads>>>(nthreads, ++iDepth);
        printf("-------> nested execution depth: %d\n", iDepth);
    }
}

int main() {
    nested_hello_world<<<1, 8>>>(8, 0);
}
extern "C" __global__ void cuda_hello(){
    int i = threadIdx.x;
    printf("Hello World from GPU, thread %d!\n", i);
}
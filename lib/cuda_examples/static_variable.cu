#include <stdio.h>
#include <cuda_runtime.h>

__device__ float dev_data;

extern "C" __global__ void check_global_variable()
{
    printf("Device: the value of the global variable is %f\n", dev_data);
    dev_data +=2.0f;
}

int main(void) {
    // initialize the global variable
    float value = 3.14f;
    cudaMemcpyToSymbol(dev_data, &value, sizeof(float));
    printf("Host: copied %f to the global variable\n", value);
    // invoke the kernel
    check_global_variable<<<1, 1>>>();
    // copy the global variable back to the host
    cudaMemcpyFromSymbol(&value, dev_data, sizeof(float));
    printf("Host: the value changed by the kernel to %f\n", value);
    cudaDeviceReset();
    return EXIT_SUCCESS;
}
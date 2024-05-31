#include <cuda_runtime.h>

#include "../cuda_common/cuda_check.h"

extern "C" void cuda_device_reset()
{
    CUDA_CHECK(cudaDeviceReset());
}

extern "C" void cuda_set_device()
{
    CUDA_CHECK(cudaSetDevice(0));
}
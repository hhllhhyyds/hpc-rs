#include <cuda_runtime.h>

#include "../cuda_common/cuda_check.h"

extern "C" void cuda_malloc(void **dev_ptr, size_t size)
{
    CUDA_CHECK(cudaMalloc(dev_ptr, size));
}

extern "C" void cuda_free(void *dev_ptr)
{
    CUDA_CHECK(cudaFree(dev_ptr));
}

extern "C" void cuda_memcpy_htod(void *host_ptr, void *dev_ptr, size_t size)
{
    CUDA_CHECK(cudaMemcpy(dev_ptr, host_ptr, size, cudaMemcpyHostToDevice));
}

extern "C" void cuda_memcpy_dtoh(void *dev_ptr, void *host_ptr, size_t size)
{
    CUDA_CHECK(cudaMemcpy(host_ptr, dev_ptr, size, cudaMemcpyDeviceToHost));
}
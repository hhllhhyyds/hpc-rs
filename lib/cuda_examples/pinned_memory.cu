#include <stdio.h>
#include <cuda_runtime.h>

void init_data(float *array, size_t size) 
{
    for (size_t i = 0; i < size; i++)
    {
        array[i] = (float)i;
    }
}

void init_zeros(float *array, size_t size)
{
    memset(array, 0, size * sizeof(float));
}

bool array_eq(float *a, float *b, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        float diff = a[i] - b[i];
        float eps = 1e-6;
        if (diff > eps || diff < -eps)
        {
            return false;
        }
    }
    return true;
}

void host_sum_array(float *a, float *b, float *c, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
    }
}

__global__ void add_array(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    } 
}

__global__ void add_array_zero_copy(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    } 
}

int main(int argc, char* argv[])
{
    // set up device
    int dev = 0;
    cudaSetDevice(dev);

    // get device information
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    // check if support mapped memory
    if (!deviceProp.canMapHostMemory) 
    {
        printf("Device %d does not support mapping CPU host memory!\n", dev);
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }

    // set up date size of vectors
    int ipower = 10;
    if (argc > 1) 
    {
        ipower = atoi(argv[1]);
    }
    size_t n_elem = 1 << ipower;
    size_t n_bytes = n_elem * sizeof(float);
    if (ipower < 18) 
    {
        printf("Vector size %d power %d nbytes %3.0f KB\n", n_elem, ipower, (float)n_bytes / (1024.0f));
    } else {
        printf("Vector size %d power %d nbytes %3.0f MB\n", n_elem, ipower, (float)n_bytes / (1024.0f * 1024.0f));
    }

    float *h_a, *h_b, *h_ref, *d_ref;
    h_a = (float*)malloc(n_bytes);
    h_b = (float*)malloc(n_bytes);
    h_ref = (float*)malloc(n_bytes);
    d_ref = (float*)malloc(n_bytes);
    init_data(h_a, n_elem);
    init_data(h_b, n_elem);
    init_zeros(h_ref, n_elem);
    init_zeros(d_ref, n_elem);
    host_sum_array(h_a, h_b, h_ref, n_elem);

    float *d_a, *d_b, *d_c;
    cudaMalloc((float**)(&d_a), n_bytes);
    cudaMalloc((float**)(&d_b), n_bytes);
    cudaMalloc((float**)(&d_c), n_bytes);
    cudaMemcpy(d_a, h_a, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n_bytes, cudaMemcpyHostToDevice);
    int i_len = 512;
    dim3 block(i_len);
    dim3 grid((n_elem + block.x - 1) / block.x);
    add_array<<<grid, block>>>(d_a, d_b, d_c, n_elem);
    cudaMemcpy(d_ref, d_c, n_bytes, cudaMemcpyDeviceToHost);
    bool arr_eq = array_eq(h_ref, d_ref, n_elem);
    if (!arr_eq)
    {
        printf("Host ref != Device ref\n");
        exit(1);
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);

    cudaMalloc((float**)(&d_c), n_bytes);
    cudaMallocHost((void**)(&h_a), n_bytes, cudaHostAllocMapped);
    cudaMallocHost((void**)(&h_b), n_bytes, cudaHostAllocMapped);
    init_data(h_a, n_elem);
    init_data(h_b, n_elem);
    init_zeros(h_ref, n_elem);
    init_zeros(d_ref, n_elem);
    host_sum_array(h_a, h_b, h_ref, n_elem);
    add_array_zero_copy<<<grid, block>>>(h_a, h_b, d_c, n_elem);
    cudaMemcpy(d_ref, d_c, n_bytes, cudaMemcpyDeviceToHost);
    arr_eq = array_eq(h_ref, d_ref, n_elem);
    if (!arr_eq)
    {
        printf("Host ref != Device ref\n");
        exit(1);
    }

    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);

    free(h_ref);
    free(d_ref);

    cudaDeviceReset();
    return EXIT_SUCCESS;
}
extern "C" __global__ void reduce_neighbored_1(int *g_idata, int *g_odata, unsigned int n) 
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;
    int idx = tid + blockIdx.x * blockDim.x;
    // boundary check
    if (idx >= n) 
    {
        return;
    };
    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) 
    {
        if ((tid % (2 * stride)) == 0) 
        {
            idata[tid] += idata[tid + stride];
        }
        // synchronize within block
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) 
    {
        g_odata[blockIdx.x] = idata[0];
    }
}

extern "C" __global__ void reduce_neighbored_2(int *g_idata, int *g_odata, unsigned int n) 
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;
    // boundary check
    if (idx >= n) 
    {
        return;
    };
    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) 
    {
        // convert tid into local array index
        int index = 2 * stride * tid;
        if (index < blockDim.x) 
        {
            idata[index] += idata[index + stride];
        }
        // synchronize within threadblock
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) 
    {
        g_odata[blockIdx.x] = idata[0];
    }
}
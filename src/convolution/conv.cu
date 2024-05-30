#include <cuda_runtime.h>

__global__ void
conv_2d_basic_kernel(const float *in, float *out, int width, int height, const float *filter, int r)
{
    const int out_xi = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_yi = blockIdx.y * blockDim.y + threadIdx.y;

    const int filter_edge = 2 * r + 1;

    float value = 0.0;

    for (int i = 0; i < filter_edge; ++i)
    {
        for (int j = 0; j < filter_edge; ++j)
        {
            int in_xi = out_xi - r + i;
            int in_yi = out_yi - r + j;
            if (in_xi >= 0 && in_xi < width && in_yi >= 0 && in_yi < height)
            {
                value += filter[j * filter_edge + i] * in[in_yi * width + in_xi];
            }
        }
    }

    out[out_yi * width + out_xi] = value;
}

/// @param input device pointer
/// @param out device pointer
/// @param filter device pointer
extern "C" void conv_2d_basic(const float *input, float *out, const int width, const int height, const float *filter, const int r)
{
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    conv_2d_basic_kernel<<<grid, block>>>(input, out, width, height, filter, r);
}
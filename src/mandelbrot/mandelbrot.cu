#include "../cuda_common/cuda_check.h"

struct C_MandelbrotGenConfig
{
    double x_range_start;
    double x_range_end;
    double y_range_start;
    double y_range_end;
    int x_pixel_count;
    int y_pixel_count;
    double diverge_limit;
    int iter_count_limit;
};

// TODO: use constant memory to store config
__global__ void gen_mandelbrot_set_kernel(unsigned int *iter_count, const struct C_MandelbrotGenConfig config)
{
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    int yi = blockIdx.y * blockDim.y + threadIdx.y;

    if (xi < config.x_pixel_count && yi < config.y_pixel_count)
    {
        const double c_x = config.x_range_start + (config.x_range_end - config.x_range_start) * xi / config.x_pixel_count;
        const double c_y = config.y_range_start + (config.y_range_end - config.y_range_start) * yi / config.y_pixel_count;

        double z_x = c_x;
        double z_y = c_y;

        const int pos = yi * config.x_pixel_count + xi;
        iter_count[pos] = config.iter_count_limit;

        const double limit = config.diverge_limit * config.diverge_limit;
        for (int count = 0; count < config.iter_count_limit; ++count)
        {
            double re = z_x * z_x - z_y * z_y + c_x;
            double im = 2.0 * z_x * z_y + c_y;
            z_x = re;
            z_y = im;
            if ((z_x * z_x + z_y * z_y) > limit)
            {
                iter_count[pos] = count;
                break;
            }
        }
    }
}

extern "C" void gen_mandelbrot_set(unsigned int *set, const struct C_MandelbrotGenConfig *config)
{
    const struct C_MandelbrotGenConfig conf = *config;

    dim3 block(32, 32);
    dim3 grid((conf.x_pixel_count + block.x - 1) / block.x, (conf.y_pixel_count + block.y - 1) / block.y);

    const int data_size = conf.x_pixel_count * conf.y_pixel_count * sizeof(unsigned int);

    unsigned int *d_set;
    CUDA_CHECK(cudaMalloc((void **)&d_set, data_size));

    gen_mandelbrot_set_kernel<<<grid, block>>>(d_set, conf);

    CUDA_CHECK(cudaMemcpy(set, d_set, data_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_set));
}
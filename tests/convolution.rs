mod mandelbrot;

use hpc_rs::{
    convolution::{
        const_filter_size, conv_2d_basic, conv_2d_constant_filter, image_process::RgbChannels,
    },
    memory::CudaDevMemory,
};
use mandelbrot::gpu_gen_mandelbrot;

#[test]
fn test_gpu_conv_2d_basic() {
    let width = 9999;
    let height = 10000;
    let r = 3;
    let filter_size = (2 * r + 1) * (2 * r + 1);
    let filter = vec![0.5_f32; filter_size];
    let input = vec![0f32; width * height];
    let mut output = vec![1f32; width * height];
    let dev_in = CudaDevMemory::from_host(&input);
    let mut dev_out = CudaDevMemory::malloc(width * height * std::mem::size_of::<f32>());
    let dev_filter = CudaDevMemory::from_host(&filter);
    conv_2d_basic(&dev_in, &mut dev_out, width, height, &dev_filter, r);
    dev_out.copy_to_host(&mut output);
    for x in output {
        assert!(x == 0.0);
    }
}

#[test]
fn test_gpu_conv_2d_constant_filter() {
    let width = 9999;
    let height = 10000;
    let filter = vec![0.5_f32; const_filter_size()];
    let input = vec![0f32; width * height];
    let mut output = vec![1f32; width * height];
    let dev_in = CudaDevMemory::from_host(&input);
    let mut dev_out = CudaDevMemory::malloc(width * height * std::mem::size_of::<f32>());
    let dev_filter = CudaDevMemory::from_host(&filter);
    conv_2d_constant_filter(&dev_in, &mut dev_out, width, height, &dev_filter);
    dev_out.copy_to_host(&mut output);
    for x in output {
        assert!(x == 0.0);
    }
}

#[test]
fn test_load_save_image() {
    let image_path = std::path::Path::new("target").join("gpu_mandelbrot_1001.png");
    if !std::path::Path::exists(&image_path) {
        gpu_gen_mandelbrot(1001);
    }
    assert!(std::path::Path::exists(&image_path));
    let rgb = RgbChannels::load_from(&image_path).unwrap();

    let out_path = std::path::Path::new("target").join("gpu_mandelbrot_1001_out.png");
    rgb.save_to(&out_path).unwrap();
    assert!(std::path::Path::exists(&out_path));
}

#[test]
fn test_blur_mandelbrot_basic() {
    let image_path = std::path::Path::new("target").join("gpu_mandelbrot_1234.png");
    if !std::path::Path::exists(&image_path) {
        gpu_gen_mandelbrot(1234);
    }
    assert!(std::path::Path::exists(&image_path));
    let mut rgb = RgbChannels::load_from(&image_path).unwrap();

    let start = std::time::Instant::now();
    rgb.blur_basic();
    let end = std::time::Instant::now();
    let duration = end - start;
    println!(
        "time used in image blur = {} ms",
        duration.as_nanos() as f32 / 1000000.
    );

    let out_path = std::path::Path::new("target").join("gpu_mandelbrot_1234_blured_basic_out.png");
    rgb.save_to(&out_path).unwrap();
    assert!(std::path::Path::exists(&out_path));
}

#[test]
fn test_blur_mandelbrot_constant_filter() {
    let image_path = std::path::Path::new("target").join("gpu_mandelbrot_2345.png");
    if !std::path::Path::exists(&image_path) {
        gpu_gen_mandelbrot(2345);
    }
    assert!(std::path::Path::exists(&image_path));
    let mut rgb = RgbChannels::load_from(&image_path).unwrap();

    let start = std::time::Instant::now();
    rgb.blur_constant_filter();
    let end = std::time::Instant::now();
    let duration = end - start;
    println!(
        "time used in image blur = {} ms",
        duration.as_nanos() as f32 / 1000000.
    );

    let out_path =
        std::path::Path::new("target").join("gpu_mandelbrot_2345_blured_constant_filter.png");
    rgb.save_to(&out_path).unwrap();
    assert!(std::path::Path::exists(&out_path));
}

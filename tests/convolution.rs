mod mandelbrot;

use hpc_rs::convolution::image_process::*;
use mandelbrot::gpu_gen_mandelbrot_3000;

#[test]
fn test_load_save_image() {
    let image_path = std::path::Path::new("target").join("gpu_mandelbrot_3000.png");
    if !std::path::Path::exists(&image_path) {
        gpu_gen_mandelbrot_3000();
    }
    assert!(std::path::Path::exists(&image_path));
    let rgb = load_image(&image_path).unwrap();

    let out_path = std::path::Path::new("target").join("gpu_mandelbrot_3000_out.png");
    save_image(&rgb, &out_path).unwrap();
    assert!(std::path::Path::exists(&out_path));
}

#[test]
fn test_blur_mandelbrot_basic() {
    let image_path = std::path::Path::new("target").join("gpu_mandelbrot_3000.png");
    if !std::path::Path::exists(&image_path) {
        gpu_gen_mandelbrot_3000();
    }
    assert!(std::path::Path::exists(&image_path));
    let mut rgb = load_image(&image_path).unwrap();
    rgb.blur_basic();
    let out_path = std::path::Path::new("target").join("gpu_mandelbrot_3000_blured_basic_out.png");
    save_image(&rgb, &out_path).unwrap();
    assert!(std::path::Path::exists(&out_path));
}

#[test]
fn test_blur_mandelbrot_constant_filter() {
    let image_path = std::path::Path::new("target").join("gpu_mandelbrot_3000.png");
    if !std::path::Path::exists(&image_path) {
        gpu_gen_mandelbrot_3000();
    }
    assert!(std::path::Path::exists(&image_path));
    let mut rgb = load_image(&image_path).unwrap();
    rgb.blur_constant_filter();
    let out_path =
        std::path::Path::new("target").join("gpu_mandelbrot_3000_blured_constant_filter.png");
    save_image(&rgb, &out_path).unwrap();
    assert!(std::path::Path::exists(&out_path));
}

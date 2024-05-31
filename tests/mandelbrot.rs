use hpc_rs::mandelbrot::MandelbrotGenConfig;
use serial_test::serial;

fn iter_count_to_rgb(iter_count: &[u32]) -> Vec<[u8; 3]> {
    let max_iter = iter_count.iter().max().unwrap();
    iter_count
        .iter()
        .map(|ic| color(*ic as f32 / *max_iter as f32))
        .collect()
}

fn color(t: f32) -> [u8; 3] {
    [(255.0 * t) as u8, (255.0 * t) as u8, (255.0 * t) as u8]
}

fn config(x_pixel_count: usize) -> MandelbrotGenConfig {
    let ratio = 1080_f64 / 1920_f64;
    let y_pixel_count = (ratio as f32 * x_pixel_count as f32) as usize;
    let iter_count_limit = 70;
    MandelbrotGenConfig {
        x_range: -2.5..1.5,
        y_range: -2.0 * ratio..2.0 * ratio,
        x_pixel_count,
        y_pixel_count,
        diverge_limit: 100.,
        iter_count_limit,
    }
}

fn iter_count_to_image(count: &[u32], config: &MandelbrotGenConfig, path: &str) {
    let color = iter_count_to_rgb(count);
    let img = image::RgbImage::from_vec(
        config.x_pixel_count as u32,
        config.y_pixel_count as u32,
        color.concat(),
    )
    .unwrap();
    img.save_with_format(path, image::ImageFormat::Png).unwrap();
}

fn cpu_gen_mandelbrot(x_pixel_count: usize) {
    let config = config(x_pixel_count);
    let start = std::time::Instant::now();
    let set = config.cpu_generate_set();
    let end = std::time::Instant::now();
    let duration = end - start;
    println!(
        "time used in cpu generate mandelbrot with dim x {} y {} = {} ms",
        config.x_pixel_count,
        config.y_pixel_count,
        duration.as_nanos() / 1000000
    );
    iter_count_to_image(
        &set,
        &config,
        &format!("target/cpu_mandelbrot_{x_pixel_count}.png"),
    );
}

pub fn gpu_gen_mandelbrot(x_pixel_count: usize) {
    let config = config(x_pixel_count);
    let start = std::time::Instant::now();
    let set = config.gpu_generate_set();
    let end = std::time::Instant::now();
    let duration = end - start;
    println!(
        "time used in gpu generate mandelbrot with dim x {} y {} = {} ms",
        config.x_pixel_count,
        config.y_pixel_count,
        duration.as_nanos() / 1000000
    );
    iter_count_to_image(
        &set,
        &config,
        &format!("target/gpu_mandelbrot_{x_pixel_count}.png"),
    );
}

#[test]
#[serial]
#[ignore = "manual"]
fn cpu_gen_mandelbrot_7200() {
    hpc_rs::device::cuda_device_reset();
    hpc_rs::device::cuda_set_device();
    cpu_gen_mandelbrot(7200);
    hpc_rs::device::cuda_device_reset();
}

#[test]
#[serial]
#[ignore = "manual"]
fn gpu_gen_mandelbrot_7200() {
    hpc_rs::device::cuda_device_reset();
    hpc_rs::device::cuda_set_device();
    gpu_gen_mandelbrot(7200);
    hpc_rs::device::cuda_device_reset();
}

#[test]
#[serial]
fn cpu_gen_mandelbrot_500() {
    hpc_rs::device::cuda_device_reset();
    hpc_rs::device::cuda_set_device();
    cpu_gen_mandelbrot(500);
    hpc_rs::device::cuda_device_reset();
}

#[test]
#[serial]
fn gpu_gen_mandelbrot_3000() {
    hpc_rs::device::cuda_device_reset();
    hpc_rs::device::cuda_set_device();
    gpu_gen_mandelbrot(3000);
    hpc_rs::device::cuda_device_reset();
}

#[test]
#[serial]
fn test_cpu_gpu_result_eq() {
    hpc_rs::device::cuda_device_reset();
    hpc_rs::device::cuda_set_device();
    let config = config(500);

    let set_cpu = config.cpu_generate_set();
    hpc_rs::device::cuda_device_reset();
    hpc_rs::device::cuda_set_device();
    let set_gpu = config.gpu_generate_set();

    for (a, b) in set_cpu.iter().zip(set_gpu.iter()) {
        assert_eq!(a, b, "a = {a}, b = {b}")
    }
    hpc_rs::device::cuda_device_reset();
}

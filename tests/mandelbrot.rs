use hpc_rs::mandelbrot::{
    binding::{gen_mandelbrot_set, CMandelbrotGenConfig},
    MandelbrotGenConfig,
};

fn iter_count_to_rgb(iter_count: &[u32]) -> Vec<[u8; 3]> {
    let mut max_iter = 0;
    for x in iter_count {
        if max_iter < *x {
            max_iter = *x;
        }
    }
    iter_count
        .iter()
        .map(|ic| color(*ic as f32 / max_iter as f32))
        .collect()
}

fn color(t: f32) -> [u8; 3] {
    [(255.0 * t) as u8, (255.0 * t) as u8, (255.0 * t) as u8]
}

fn config() -> MandelbrotGenConfig {
    let x_count = 7200;
    let ratio = 1080 as f64 / 1920 as f64;
    let y_count = (ratio as f32 * x_count as f32) as usize;
    let iter_count_limit = 70;
    MandelbrotGenConfig {
        x_range: -2.5..1.5,
        y_range: -2.0 * ratio..2.0 * ratio,
        x_pixel_count: x_count,
        y_pixel_count: y_count,
        diverge_limit: 100.,
        iter_count_limit,
    }
}

fn iter_count_to_image(count: &[u32], config: &MandelbrotGenConfig, path: &str) {
    let color = iter_count_to_rgb(&count);
    let img = image::RgbImage::from_vec(
        config.x_pixel_count as u32,
        config.y_pixel_count as u32,
        color.concat(),
    )
    .unwrap();
    img.save_with_format(path, image::ImageFormat::Png).unwrap();
}

#[test]
#[ignore = "manual"]
fn test_gen_mandelbrot() {
    let start = std::time::Instant::now();
    let set = config().generate_set();
    let end = std::time::Instant::now();
    let duration = end - start;
    println!(
        "time used in cpu generate mandelbrot = {} ms",
        duration.as_nanos() / 1000000
    );
    iter_count_to_image(&set, &config(), "target/mandelbrot.png");
}

#[test]
#[ignore = "manual"]
fn test_cuda_gen_mandelbrot() {
    let config = config();
    let c_config = CMandelbrotGenConfig::from(config.clone());
    let mut set = vec![0u32; config.pixel_count()];
    let start = std::time::Instant::now();
    unsafe { gen_mandelbrot_set(set.as_mut_ptr(), &c_config) };
    let end = std::time::Instant::now();
    let duration = end - start;
    println!(
        "time used in gpu generate mandelbrot = {} ms",
        duration.as_nanos() / 1000000
    );
    iter_count_to_image(&set, &config, "target/gpu_mandelbrot.png");
}

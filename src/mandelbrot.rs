use std::ops::Range;

use num::complex::Complex64;

pub struct MandelbrotGenConfig {
    pub x_range: Range<f64>,
    pub y_range: Range<f64>,
    pub x_pixel_count: usize,
    pub y_pixel_count: usize,
    pub diverge_limit: f64,
    pub iter_count_limit: usize,
}

impl MandelbrotGenConfig {
    fn pixel_count(&self) -> usize {
        self.x_pixel_count * self.y_pixel_count
    }
    fn pixel_xy_to_1d(&self, x: usize, y: usize) -> usize {
        y * self.x_pixel_count + x
    }
    fn pixel_xy_to_coord(&self, x: usize, y: usize) -> Complex64 {
        let x_coord = self.x_range.start
            + (self.x_range.end - self.x_range.start) * x as f64 / self.x_pixel_count as f64;
        let y_coord = self.y_range.start
            + (self.y_range.end - self.y_range.start) * y as f64 / self.y_pixel_count as f64;
        Complex64::new(x_coord, y_coord)
    }
    fn coord_iter_once(z_n: Complex64, c: Complex64) -> Complex64 {
        z_n * z_n + c
    }

    pub fn generate_set(&self) -> Vec<u32> {
        let n = self.pixel_count();
        let mut out = vec![self.iter_count_limit as u32; n];

        for y in 0..self.y_pixel_count {
            for x in 0..self.x_pixel_count {
                let c = self.pixel_xy_to_coord(x, y);
                let limit = self.diverge_limit * self.diverge_limit;
                let mut z = c;

                for i in 0..self.iter_count_limit {
                    z = Self::coord_iter_once(z, c);
                    if z.norm_sqr() > limit {
                        out[self.pixel_xy_to_1d(x, y)] = i as u32 + 1;
                        break;
                    }
                }
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    #[ignore = "manual"]
    fn test_gen_mandelbrot() {
        let x_count = 7200;
        let ratio = 1080 as f64 / 1920 as f64;
        let y_count = (ratio as f32 * x_count as f32) as usize;
        let iter_count_limit = 70;
        let config = MandelbrotGenConfig {
            x_range: -2.5..1.5,
            y_range: -2.0 * ratio..2.0 * ratio,
            x_pixel_count: x_count,
            y_pixel_count: y_count,
            diverge_limit: 100.,
            iter_count_limit,
        };
        let set = config.generate_set();
        let color = iter_count_to_rgb(&set);
        let img =
            image::RgbImage::from_vec(x_count as u32, y_count as u32, color.concat()).unwrap();
        img.save_with_format("target/mandelbrot.png", image::ImageFormat::Png)
            .unwrap();
    }
}

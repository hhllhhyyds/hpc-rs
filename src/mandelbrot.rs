use std::ops::Range;

use num::complex::Complex64;

mod binding;

#[derive(Clone, Debug)]
pub struct MandelbrotGenConfig {
    pub x_range: Range<f64>,
    pub y_range: Range<f64>,
    pub x_pixel_count: usize,
    pub y_pixel_count: usize,
    pub diverge_limit: f64,
    pub iter_count_limit: usize,
}

impl MandelbrotGenConfig {
    pub fn pixel_count(&self) -> usize {
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

    pub fn cpu_generate_set(&self) -> Vec<u32> {
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

    pub fn gpu_generate_set(&self) -> Vec<u32> {
        let n = self.pixel_count();
        let mut out = vec![self.iter_count_limit as u32; n];

        unsafe {
            binding::gen_mandelbrot_set(
                out.as_mut_ptr(),
                &binding::CMandelbrotGenConfig::from(self.clone()),
            )
        };

        out
    }
}

impl From<MandelbrotGenConfig> for binding::CMandelbrotGenConfig {
    fn from(value: MandelbrotGenConfig) -> Self {
        Self {
            x_range_start: value.x_range.start,
            x_range_end: value.x_range.end,
            y_range_start: value.y_range.start,
            y_range_end: value.y_range.end,
            x_pixel_count: value.x_pixel_count as i32,
            y_pixel_count: value.y_pixel_count as i32,
            diverge_limit: value.diverge_limit,
            iter_count_limit: value.iter_count_limit as i32,
        }
    }
}

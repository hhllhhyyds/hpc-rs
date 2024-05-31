use image::{ImageBuffer, ImageResult, Rgb};

use crate::memory::CudaDevMemory;

use super::{const_filter_size, conv_2d_basic, conv_2d_constant_filter};

#[derive(Default, Clone)]
pub struct RgbChannels {
    pub width: u32,
    pub height: u32,
    pub r: Vec<f32>,
    pub g: Vec<f32>,
    pub b: Vec<f32>,
}

impl RgbChannels {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn assembly(&self) -> Vec<u8> {
        let mut ret = vec![];
        for i in 0..self.r.len() {
            ret.push((self.r[i] * 255.) as u8);
            ret.push((self.g[i] * 255.) as u8);
            ret.push((self.b[i] * 255.) as u8);
        }
        ret
    }

    pub fn blur_basic(&mut self) {
        let r = 3;
        let edge = 2 * r + 1;
        let filter = vec![1.0 / (edge * edge) as f32; edge * edge];

        let size = (self.width * self.height) as usize;

        let dev_in = CudaDevMemory::malloc(size * std::mem::size_of::<f32>());
        let mut dev_out = CudaDevMemory::malloc(size * std::mem::size_of::<f32>());
        let dev_filter = CudaDevMemory::from_host(&filter);

        for channel in [&mut self.r, &mut self.g, &mut self.b] {
            dev_in.copy_from_host(channel);
            conv_2d_basic(
                &dev_in,
                &mut dev_out,
                self.width as usize,
                self.height as usize,
                &dev_filter,
                r,
            );
            dev_out.copy_to_host(channel);
        }
    }

    pub fn blur_constant_filter(&mut self) {
        let filter = vec![1.0 / super::const_filter_size() as f32; const_filter_size()];

        let size = (self.width * self.height) as usize;

        let dev_in = CudaDevMemory::malloc(size * std::mem::size_of::<f32>());
        let mut dev_out = CudaDevMemory::malloc(size * std::mem::size_of::<f32>());
        let dev_filter = CudaDevMemory::from_host(&filter);

        // Bug: only first channel is blured
        for channel in [&mut self.g, &mut self.r, &mut self.b] {
            dev_in.copy_from_host(channel);
            conv_2d_constant_filter(
                &dev_in,
                &mut dev_out,
                self.width as usize,
                self.height as usize,
                &dev_filter,
            );
            dev_out.copy_to_host(channel);
        }
    }

    pub fn load_from(path: impl AsRef<std::path::Path>) -> ImageResult<Self> {
        load_image(path)
    }

    pub fn save_to(&self, path: impl AsRef<std::path::Path>) -> ImageResult<()> {
        save_image(self, path)
    }
}

fn load_image(path: impl AsRef<std::path::Path>) -> ImageResult<RgbChannels> {
    let img = image::io::Reader::open(path)?.decode()?.to_rgb32f();
    let (width, height) = (img.width(), img.height());
    let size = width * height;
    let mut rgb = RgbChannels::new();
    for pixel in img.pixels() {
        rgb.r.push(pixel[0]);
        rgb.g.push(pixel[1]);
        rgb.b.push(pixel[2]);
    }
    rgb.width = width;
    rgb.height = height;

    assert!(rgb.r.len() == size as usize);
    assert!(rgb.g.len() == size as usize);
    assert!(rgb.b.len() == size as usize);

    Ok(rgb)
}

fn save_image(rgb: &RgbChannels, path: impl AsRef<std::path::Path>) -> ImageResult<()> {
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        image::ImageBuffer::from_vec(rgb.width, rgb.height, rgb.assembly())
            .expect("failed to construct image buffer");
    img.save(path)?;
    Ok(())
}

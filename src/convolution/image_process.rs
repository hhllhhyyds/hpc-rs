use std::os::raw;

use image::{ImageBuffer, ImageResult, Rgb};

use crate::memory::CudaDevMemory;

use super::{binding, const_filter_size};

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

        let dev_out = CudaDevMemory::new(size * std::mem::size_of::<f32>());
        let dev_filter = CudaDevMemory::from_host(&filter);

        let dev_in = CudaDevMemory::from_host(&self.r);
        unsafe {
            binding::conv_2d_basic(
                dev_in.dev_ptr() as *const raw::c_float,
                dev_out.dev_ptr() as *mut raw::c_float,
                self.width as i32,
                self.height as i32,
                dev_filter.dev_ptr() as *const raw::c_float,
                r as i32,
            )
        };
        dev_out.copy_to_host(&mut self.r);

        dev_in.copy_from_host(&self.g);
        unsafe {
            binding::conv_2d_basic(
                dev_in.dev_ptr() as *const raw::c_float,
                dev_out.dev_ptr() as *mut raw::c_float,
                self.width as i32,
                self.height as i32,
                dev_filter.dev_ptr() as *const raw::c_float,
                r as i32,
            )
        };
        dev_out.copy_to_host(&mut self.g);

        dev_in.copy_from_host(&self.b);
        unsafe {
            binding::conv_2d_basic(
                dev_in.dev_ptr() as *const raw::c_float,
                dev_out.dev_ptr() as *mut raw::c_float,
                self.width as i32,
                self.height as i32,
                dev_filter.dev_ptr() as *const raw::c_float,
                r as i32,
            )
        };
        dev_out.copy_to_host(&mut self.b);
    }

    pub fn blur_constant_filter(&mut self) {
        let filter = vec![1.0 / super::const_filter_size() as f32; const_filter_size()];

        let size = (self.width * self.height) as usize;

        let dev_out = CudaDevMemory::new(size * std::mem::size_of::<f32>());

        let dev_in = CudaDevMemory::from_host(&self.r);
        unsafe {
            binding::conv_2d_constant_filter(
                dev_in.dev_ptr() as *const raw::c_float,
                dev_out.dev_ptr() as *mut raw::c_float,
                self.width as i32,
                self.height as i32,
                filter.as_ptr(),
            )
        };
        dev_out.copy_to_host(&mut self.r);

        dev_in.copy_from_host(&self.g);
        unsafe {
            binding::conv_2d_constant_filter(
                dev_in.dev_ptr() as *const raw::c_float,
                dev_out.dev_ptr() as *mut raw::c_float,
                self.width as i32,
                self.height as i32,
                filter.as_ptr(),
            )
        };
        dev_out.copy_to_host(&mut self.g);

        dev_in.copy_from_host(&self.b);
        unsafe {
            binding::conv_2d_constant_filter(
                dev_in.dev_ptr() as *const raw::c_float,
                dev_out.dev_ptr() as *mut raw::c_float,
                self.width as i32,
                self.height as i32,
                filter.as_ptr(),
            )
        };
        dev_out.copy_to_host(&mut self.b);
    }
}

pub fn load_image(path: impl AsRef<std::path::Path>) -> ImageResult<RgbChannels> {
    let img = image::io::Reader::open(path)?.decode()?.to_rgb32f();
    let (width, height) = (img.width(), img.height());
    let size = width * height;
    let mut rgb = RgbChannels::new();
    for pixel in img.pixels() {
        rgb.r.push(pixel[0]);
        rgb.g.push(pixel[1]);
        rgb.b.push(pixel[2]);
        // println!("rgb = {}, {}, {}", pixel[0], pixel[1], pixel[2]);
    }
    rgb.width = width;
    rgb.height = height;

    assert!(rgb.r.len() == size as usize);
    assert!(rgb.g.len() == size as usize);
    assert!(rgb.b.len() == size as usize);

    Ok(rgb)
}

pub fn save_image(rgb: &RgbChannels, path: impl AsRef<std::path::Path>) -> ImageResult<()> {
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        image::ImageBuffer::from_vec(rgb.width, rgb.height, rgb.assembly())
            .expect("failed to construct image buffer");
    img.save(path)?;
    Ok(())
}

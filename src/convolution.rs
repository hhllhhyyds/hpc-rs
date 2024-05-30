use std::os::raw;

use crate::memory::CudaDevMemory;

mod binding;

pub mod image_process;

#[allow(unused)]
fn simple_gpu_conv_2d(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    filter: &[f32],
    r: usize,
) {
    let size = width * height;
    assert!(input.len() == size);
    assert!(output.len() == size);
    let filter_size = (2 * r + 1) * (2 * r + 1);
    assert!(filter_size == filter.len());

    let dev_in = CudaDevMemory::from_host(input);
    let dev_out = CudaDevMemory::new(size * std::mem::size_of::<f32>());
    let dev_filter = CudaDevMemory::from_host(filter);

    unsafe {
        binding::conv_2d_basic(
            dev_in.dev_ptr() as *const raw::c_float,
            dev_out.dev_ptr() as *mut raw::c_float,
            width as i32,
            height as i32,
            dev_filter.dev_ptr() as *const raw::c_float,
            r as i32,
        )
    };

    dev_out.copy_to_host(output);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_conv_2d() {
        let width = 9999;
        let height = 10000;
        let r = 3;
        let filter_size = (2 * r + 1) * (2 * r + 1);
        let filter = vec![0.5_f32; filter_size];
        let input = vec![0f32; width * height];
        let mut output = vec![1f32; width * height];
        simple_gpu_conv_2d(&input, &mut output, width, height, &filter, r);
        for x in output {
            assert!(x == 0.0);
        }
    }
}

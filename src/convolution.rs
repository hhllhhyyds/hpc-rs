use std::os::raw;

use crate::memory::CudaDevMemory;

mod binding;
pub mod image_process;

pub const CONSTANT_FILTER_RADIUS: usize = 3;
pub const fn const_filter_size() -> usize {
    (CONSTANT_FILTER_RADIUS * 2 + 1) * (CONSTANT_FILTER_RADIUS * 2 + 1)
}

pub fn conv_2d_basic(
    input: &CudaDevMemory,
    output: &mut CudaDevMemory,
    width: usize,
    height: usize,
    filter: &CudaDevMemory,
    r: usize,
) {
    let data_size = width * height * std::mem::size_of::<raw::c_float>();
    assert!(input.size_in_bytes() == data_size);
    assert!(output.size_in_bytes() == data_size);

    unsafe {
        binding::conv_2d_basic(
            input.as_ptr::<raw::c_float>(),
            output.as_mut_ptr::<raw::c_float>(),
            width as i32,
            height as i32,
            filter.as_ptr::<raw::c_float>(),
            r as i32,
        )
    };
}

pub fn conv_2d_constant_filter(
    input: &CudaDevMemory,
    output: &mut CudaDevMemory,
    width: usize,
    height: usize,
    filter: &CudaDevMemory,
) {
    let data_size = width * height * std::mem::size_of::<raw::c_float>();
    assert!(input.size_in_bytes() == data_size);
    assert!(output.size_in_bytes() == data_size);
    assert!(filter.size_in_bytes() == const_filter_size() * std::mem::size_of::<raw::c_float>());

    unsafe {
        binding::conv_2d_constant_filter(
            input.as_ptr::<raw::c_float>(),
            output.as_mut_ptr::<raw::c_float>(),
            width as i32,
            height as i32,
            filter.as_ptr::<raw::c_float>(),
        )
    };
}

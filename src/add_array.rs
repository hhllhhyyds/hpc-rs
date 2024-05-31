use crate::cuda_thread_hierarchy::CudaThreadHierarchy;
use std::ffi::{c_float, c_int};

pub fn add_array_cuda(a: &[f32], b: &[f32]) -> Vec<f32> {
    extern "C" {
        fn add_array(
            a: *const c_float,
            b: *const c_float,
            c: *mut c_float,
            N: c_int,
            grid: c_int,
            block: c_int,
        );
    }

    assert!(a.len() == b.len());

    let n = a.len();
    let mut out = vec![0.; n];

    let thread_hierarchy = CudaThreadHierarchy::dim1_default(n);
    unsafe {
        add_array(
            a.as_ptr(),
            b.as_ptr(),
            out.as_mut_ptr(),
            n as c_int,
            thread_hierarchy.grid.x as c_int,
            thread_hierarchy.block.x as c_int,
        )
    };

    out
}

use std::ffi::{c_float, c_int};

#[repr(C)]
struct Dim3 {
    pub x: c_int,
    pub y: c_int,
    pub z: c_int,
}
impl From<(usize, usize, usize)> for Dim3 {
    fn from(value: (usize, usize, usize)) -> Self {
        Self {
            x: value.0 as i32 as c_int,
            y: value.1 as i32 as c_int,
            z: value.2 as i32 as c_int,
        }
    }
}

extern "C" {
    fn add_array(
        a: *const c_float,
        b: *const c_float,
        c: *mut c_float,
        N: c_int,
        grid: Dim3,
        block: Dim3,
    );
}

pub fn add_array_cuda(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32> {
    assert!(a.len() == b.len());
    let mut out = vec![0.; a.len()];

    let block = Dim3::from((1024, 1, 1));
    let grid = Dim3::from(((a.len() + block.x as usize - 1) / block.x as usize, 1, 1));
    unsafe {
        add_array(
            a.as_ptr(),
            b.as_ptr(),
            out.as_mut_ptr(),
            a.len() as i32,
            grid,
            block,
        )
    };

    out
}

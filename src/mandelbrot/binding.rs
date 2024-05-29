#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CMandelbrotGenConfig {
    pub x_range_start: f64,
    pub x_range_end: f64,
    pub y_range_start: f64,
    pub y_range_end: f64,
    pub x_pixel_count: ::std::os::raw::c_int,
    pub y_pixel_count: ::std::os::raw::c_int,
    pub diverge_limit: f64,
    pub iter_count_limit: ::std::os::raw::c_int,
}

extern "C" {
    pub fn gen_mandelbrot_set(
        set: *mut ::std::os::raw::c_uint,
        config: *const CMandelbrotGenConfig,
    );
}

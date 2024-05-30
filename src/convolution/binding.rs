use std::os::raw;

extern "C" {
    pub fn conv_2d_basic(
        input: *const raw::c_float,
        out: *mut raw::c_float,
        width: raw::c_int,
        height: raw::c_int,
        filter: *const raw::c_float,
        r: raw::c_int,
    );
}

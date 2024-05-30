use std::os::raw::c_void;

extern "C" {
    pub fn cuda_malloc(dev_ptr: *mut *mut c_void, size: usize);
    pub fn cuda_free(dev_ptr: *const c_void);
    pub fn cuda_memcpy_htod(host_ptr: *const c_void, dev_ptr: *mut c_void, size: usize);
    pub fn cuda_memcpy_dtoh(dev_ptr: *const c_void, host_ptr: *mut c_void, size: usize);
}

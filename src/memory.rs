use std::os::raw;

mod binding;

pub struct CudaDevMemory {
    /// size in bytes
    size: usize,
    ptr: *mut raw::c_void,
}

impl CudaDevMemory {
    pub fn malloc(size: usize) -> Self {
        let mut ptr = std::ptr::null_mut();
        unsafe { binding::cuda_malloc(std::ptr::addr_of_mut!(ptr), size) };
        Self { ptr, size }
    }

    pub fn copy_from_host<T>(&self, host_data: &[T]) {
        debug_assert!(std::mem::size_of_val(host_data) == self.size);
        unsafe {
            binding::cuda_memcpy_htod(
                host_data.as_ptr() as *const raw::c_void,
                self.ptr,
                self.size,
            )
        }
    }

    pub fn from_host<T>(host_data: &[T]) -> Self {
        let size = std::mem::size_of_val(host_data);
        let memory = Self::malloc(size);
        memory.copy_from_host(host_data);
        memory
    }

    pub fn as_ptr<T>(&self) -> *const T {
        self.ptr as *const T
    }

    pub fn as_mut_ptr<T>(&self) -> *mut T {
        self.ptr as *mut T
    }

    pub fn size_in_bytes(&self) -> usize {
        self.size
    }

    pub fn copy_to_host<T>(&self, host_data: &mut [T]) {
        debug_assert!(std::mem::size_of_val(host_data) == self.size);
        unsafe {
            binding::cuda_memcpy_dtoh(
                self.ptr,
                host_data.as_mut_ptr() as *mut raw::c_void,
                self.size,
            )
        }
    }
}

impl Drop for CudaDevMemory {
    fn drop(&mut self) {
        assert!(!self.as_ptr::<raw::c_void>().is_null());
        unsafe { binding::cuda_free(self.ptr) };
        self.size = 0;
        self.ptr = std::ptr::null_mut();
    }
}

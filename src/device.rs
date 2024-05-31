mod binding;

pub fn cuda_device_reset() {
    unsafe {
        binding::cuda_device_reset();
    }
}

pub fn cuda_set_device() {
    unsafe {
        binding::cuda_set_device();
    }
}

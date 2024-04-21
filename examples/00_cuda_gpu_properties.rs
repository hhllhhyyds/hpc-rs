use cudarc::driver::{
    sys::{cuDeviceGetCount, cuDeviceGetName, cuDeviceGetProperties, CUdevprop},
    CudaDevice,
};

use std::{ffi::CStr, mem};

fn main() {
    let _ = CudaDevice::new(0).expect("Fail to initialize gpu device");
    unsafe {
        let mut count: i32 = 0;
        cuDeviceGetCount(std::ptr::addr_of_mut!(count));
        println!("Device count = {count}");

        if count > 0 {
            let device_id = 0;
            let name: *mut i8 = libc::malloc(mem::size_of::<i8>() * 100) as *mut i8;
            if name.is_null() {
                panic!("failed to allocate memory");
            }

            cuDeviceGetName(name, 100, device_id);

            let name_cstr = CStr::from_ptr(name);

            libc::free(name as *mut libc::c_void);

            let mut prop: CUdevprop = Default::default();
            cuDeviceGetProperties(std::ptr::addr_of_mut!(prop), device_id);

            println!(
                "Device index : {device_id}, Name: {:?}, Properties: {:?}",
                name_cstr, prop
            );
        }
    }
}

use std::ffi::{c_char, CStr};

use cudarc::driver::{
    result,
    sys::{cuDeviceGetCount, cuDeviceGetName, cuDeviceGetProperties, CUdevice},
    DriverError,
};

pub use cudarc::driver::sys::CUdevprop as GpuProperties;

#[derive(Debug)]
pub enum ErrorCudaInitialize {
    InitializeDriverError(DriverError),
    NoDeviceError,
    DeviceNotFoundError(usize),
}

#[inline]
pub fn cuda_initialize() -> Result<(), ErrorCudaInitialize> {
    result::init().map_err(ErrorCudaInitialize::InitializeDriverError)
}

#[derive(Debug)]
pub struct GpuInfo {
    index: usize,
    name: String,
    properties: GpuProperties,
}

impl GpuInfo {
    pub fn with_index(index: usize) -> Result<Self, ErrorCudaInitialize> {
        let device_count = {
            let mut count = 0;
            unsafe { cuDeviceGetCount(std::ptr::addr_of_mut!(count)) };
            count as usize
        };

        if device_count == 0 {
            return Err(ErrorCudaInitialize::NoDeviceError);
        } else if device_count <= index {
            return Err(ErrorCudaInitialize::DeviceNotFoundError(index));
        }

        Ok(Self {
            index,
            name: {
                let name_len = 1000;
                let mut name = vec!['\n' as c_char; name_len];
                unsafe {
                    cuDeviceGetName(name.as_mut_ptr(), name_len as i32, index as CUdevice);
                    CStr::from_ptr(name.as_ptr()).to_string_lossy().to_string()
                }
            },
            properties: {
                let mut prop = GpuProperties::default();
                unsafe { cuDeviceGetProperties(std::ptr::addr_of_mut!(prop), index as CUdevice) };
                prop
            },
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn properties(&self) -> &GpuProperties {
        &self.properties
    }
}

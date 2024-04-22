use std::ffi::{c_char, CStr};

use cudarc::driver::{
    result::DriverError,
    sys::{cuDeviceGetCount, cuDeviceGetName, cuDeviceGetProperties, CUdevice},
    CudaDevice,
};

pub use cudarc::driver::sys::CUdevprop as GpuProperties;

#[inline]
pub fn cuda_initialize() -> Result<(), DriverError> {
    let _ = CudaDevice::new(0)?;
    Ok(())
}

#[derive(Debug)]
pub struct GpuInfo {
    index: usize,
    name: String,
    properties: GpuProperties,
}

impl GpuInfo {
    pub fn with_index(index: usize) -> Self {
        assert!(
            {
                let mut count = 0;
                unsafe { cuDeviceGetCount(std::ptr::addr_of_mut!(count)) };
                count
            } > index as i32,
            "Gpu device with index = {} not available",
            index
        );

        Self {
            index,
            name: {
                let name_len = 1000;
                let mut name = vec![c_char::default(); name_len];
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
        }
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

impl Default for GpuInfo {
    fn default() -> Self {
        Self::with_index(CUdevice::default() as usize)
    }
}

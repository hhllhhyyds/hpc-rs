use std::sync::Arc;

use cudarc::{
    driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig},
    nvrtc::Ptx,
};

fn main() -> Result<(), DriverError> {
    let dev = CudaDevice::new(0)?;

    gpu_hello_world(&dev)?;

    check_index(&dev)?;

    Ok(())
}

fn gpu_hello_world(dev: &Arc<CudaDevice>) -> Result<(), DriverError> {
    let func_name = "cuda_hello";

    dev.load_ptx(
        Ptx::from_src(hpc_rs::ptx::HELLO_CUDA),
        func_name,
        &[func_name],
    )?;

    let cuda_func = dev
        .get_func(&func_name, &func_name)
        .unwrap_or_else(|| panic!("function {} not found in ptx", func_name));

    let cfg = LaunchConfig {
        block_dim: (10, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe { cuda_func.launch(cfg, (1,)) }?;

    Ok(())
}

fn check_index(dev: &Arc<CudaDevice>) -> Result<(), DriverError> {
    let module_name = "hello_cuda";
    let func_name = "check_index";

    dev.load_ptx(
        Ptx::from_src(hpc_rs::ptx::HELLO_CUDA),
        module_name,
        &[func_name],
    )?;

    let cuda_func = dev
        .get_func(module_name, func_name)
        .unwrap_or_else(|| panic!("function {} not found in ptx", func_name));

    let n_elem = 6;
    let block_dim = (3, 1, 1);
    let grid_dim = ((n_elem + block_dim.0 - 1) / block_dim.0, 1, 1);
    let cfg = LaunchConfig {
        block_dim,
        grid_dim,
        shared_mem_bytes: 0,
    };

    unsafe { cuda_func.launch(cfg, (1,)) }?;

    Ok(())
}

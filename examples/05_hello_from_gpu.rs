use cudarc::{
    driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig},
    nvrtc::Ptx,
};

fn main() -> Result<(), DriverError> {
    let dev = CudaDevice::new(0)?;

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

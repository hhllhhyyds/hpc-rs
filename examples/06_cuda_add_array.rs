use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use hpc_rs::error::Error;

fn main() -> Result<(), Error> {
    let dev = CudaDevice::new(0)?;

    dev.load_ptx(
        Ptx::from_src(hpc_rs::ptx::GPU_ADD_ARRAY),
        "gpu_add_array",
        &["add_array"],
    )?;
    let cuda_func = dev
        .get_func("gpu_add_array", "add_array")
        .expect("Can not find function in ptx");

    let n_elem = 32;
    let host_a = (0..n_elem).map(|x| x as f32).collect::<Vec<_>>();
    let host_b = host_a.clone();

    let device_a = dev.htod_copy(host_a)?;
    let device_b = dev.htod_copy(host_b)?;
    let device_c = unsafe { dev.alloc::<f32>(n_elem)? };

    let block_dim_x = n_elem as u32 / 4;
    let grid_dim_x = (n_elem as u32 + block_dim_x - 1) / block_dim_x;
    let cfg = LaunchConfig {
        block_dim: (block_dim_x, 1, 1),
        grid_dim: (grid_dim_x, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { cuda_func.launch(cfg, (&device_a, &device_b, &device_c, n_elem as i32)) }?;

    let result = dev.dtoh_sync_copy(&device_c)?;

    for (i, x) in result.into_iter().enumerate() {
        assert!(x == (i + i) as f32)
    }

    Ok(())
}

use cudarc::driver::{result, sys, CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use hpc_rs::error::Error;

fn main() -> Result<(), Error> {
    let dev = CudaDevice::new(0)?;

    let is_async = unsafe {
        result::device::get_attribute(
            0,
            sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
        )?
    } > 0;
    println!("device is_async = {is_async}");

    let start = std::time::Instant::now();
    dev.load_ptx(
        Ptx::from_src(hpc_rs::cuda::ptx::GPU_ADD_ARRAY),
        "gpu_add_array",
        &["add_array"],
    )?;
    let cuda_func = dev
        .get_func("gpu_add_array", "add_array")
        .expect("Can not find function in ptx");
    println!(
        "Time used in loading ptx and cuda functions = {:?}",
        start.elapsed()
    );

    let n_elem = 1 << 24;

    let start = std::time::Instant::now();
    let host_a = (0..n_elem).map(|x| x as f32).collect::<Vec<_>>();
    let host_b = host_a.clone();
    println!(
        "Time used in initializing input data = {:?}, vector size = {}",
        start.elapsed(),
        n_elem
    );

    let start = std::time::Instant::now();
    let device_a = dev.htod_copy(host_a)?;
    let device_b = dev.htod_copy(host_b)?;
    let device_c = unsafe { dev.alloc::<f32>(n_elem)? };
    println!(
        "Time used in allocate and copy to device data = {:?}",
        start.elapsed()
    );

    let block_dim_x = 1024;
    let grid_dim_x = (n_elem as u32 + block_dim_x - 1) / block_dim_x;
    let cfg = LaunchConfig {
        block_dim: (block_dim_x, 1, 1),
        grid_dim: (grid_dim_x, 1, 1),
        shared_mem_bytes: 0,
    };
    println!("Launch cfg = {:?}", cfg);
    let start = std::time::Instant::now();
    unsafe { cuda_func.launch(cfg, (&device_a, &device_b, &device_c, n_elem as i32)) }?;
    dev.synchronize()?;
    println!("Time used in running kernel = {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let result = dev.dtoh_sync_copy(&device_c)?;
    println!(
        "Time used in copy data back to host = {:?}",
        start.elapsed()
    );

    for (i, x) in result.into_iter().enumerate() {
        assert!(x == (i + i) as f32)
    }

    Ok(())
}

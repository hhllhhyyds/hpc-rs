use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use hpc_rs::cuda_initialize::GpuInfo;
use hpc_rs::error::Error;
use hpc_rs::ptx_loader::PtxLoader;
use rand::Rng;

fn main() -> Result<(), Error> {
    let dev = CudaDevice::new(0)?;

    let size: u32 = 1 << 24;
    let block_size: u32 = {
        let args: Vec<String> = std::env::args().collect();
        let block_size = if args.len() > 1 {
            args[1].parse().unwrap()
        } else {
            512
        };
        let gpu_info = GpuInfo::with_index(0).unwrap();
        assert!(block_size <= gpu_info.properties().maxThreadsPerBlock as u32);
        block_size
    };
    let block_dim_x = block_size;
    let (block_dim, grid_dim) = (
        (block_dim_x, 1u32, 1u32),
        ((size + block_size - 1) / block_size, 1u32, 1u32),
    );
    println!("block dim: {block_dim:?}, grid_dim: {grid_dim:?}");

    let host_idata = {
        let mut rng = rand::thread_rng();
        (0..size)
            .map(|_| rng.gen_range(0..10))
            .collect::<Vec<u32>>()
    };
    let start = std::time::Instant::now();
    let host_sum: u32 = host_idata.iter().sum();
    println!("Time used in cup sum = {:?}", start.elapsed());

    let mut device_idata = dev.htod_copy(host_idata.clone()).unwrap();
    let device_odata = dev.alloc_zeros::<u32>(grid_dim.0 as usize)?;

    let ptx_loader = PtxLoader::new(
        &dev,
        hpc_rs::ptx::REDUCE_SUM,
        "reduce_sum",
        &[
            "reduce_neighbored_1",
            "reduce_neighbored_2",
            "reduce_interleaved",
        ],
    );

    let cuda_func_1 = ptx_loader.get_func("reduce_neighbored_1").unwrap();
    let cuda_func_2 = ptx_loader.get_func("reduce_neighbored_2").unwrap();
    let cuda_func_3 = ptx_loader.get_func("reduce_interleaved").unwrap();

    let cfg = LaunchConfig {
        block_dim,
        grid_dim,
        shared_mem_bytes: 0,
    };

    let start = std::time::Instant::now();
    unsafe { cuda_func_1.launch(cfg, (&device_idata, &device_odata, size)) }?;
    dev.synchronize()?;
    println!(
        "Time used in running gpu reduce sum kernel cuda_func_1 = {:?}",
        start.elapsed()
    );
    let gpu_result = dev.dtoh_sync_copy(&device_odata).unwrap();
    let gpu_sum: u32 = gpu_result.iter().sum();
    assert!(
        host_sum == gpu_sum,
        "host_sum = {host_sum}, gpu_sum = {gpu_sum}"
    );

    dev.htod_copy_into(host_idata.clone(), &mut device_idata)?;
    let start = std::time::Instant::now();
    unsafe { cuda_func_2.launch(cfg, (&device_idata, &device_odata, size)) }?;
    dev.synchronize()?;
    println!(
        "Time used in running gpu reduce sum kernel cuda_func_2 = {:?}",
        start.elapsed()
    );
    let gpu_result = dev.dtoh_sync_copy(&device_odata).unwrap();
    let gpu_sum: u32 = gpu_result.iter().sum();
    assert!(
        host_sum == gpu_sum,
        "host_sum = {host_sum}, gpu_sum = {gpu_sum}"
    );

    dev.htod_copy_into(host_idata, &mut device_idata)?;
    let start = std::time::Instant::now();
    unsafe { cuda_func_3.launch(cfg, (&device_idata, &device_odata, size)) }?;
    dev.synchronize()?;
    println!(
        "Time used in running gpu reduce sum kernel cuda_func_3 = {:?}",
        start.elapsed()
    );
    let gpu_result = dev.dtoh_sync_copy(&device_odata).unwrap();
    let gpu_sum: u32 = gpu_result.iter().sum();
    assert!(
        host_sum == gpu_sum,
        "host_sum = {host_sum}, gpu_sum = {gpu_sum}"
    );

    Ok(())
}

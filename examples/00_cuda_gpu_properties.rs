use hpc_rs::cuda_initialize;

fn main() {
    cuda_initialize::cuda_initialize().unwrap();
    println!("{:#?}", cuda_initialize::GpuInfo::with_index(0).unwrap());
}

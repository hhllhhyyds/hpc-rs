use hpc_rs::cuda_initialize;

fn main() {
    cuda_initialize::cuda_initialize().expect("Fail to initialize cuda");
    println!("{:#?}", cuda_initialize::GpuInfo::default());
}

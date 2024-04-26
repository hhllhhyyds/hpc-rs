use std::error::Error;

use cmake::Config;

fn build_cuda_ptx() {
    const CUDA_KERNEL_PATH_PATTERN: &str = "src/cuda/kernels/*.cu";
    const PTX_STRING_PATH: &str = "src/cuda/ptx.rs";

    let builder = bindgen_cuda::Builder::default().kernel_paths_glob(CUDA_KERNEL_PATH_PATTERN);
    let bindings = builder.build_ptx().expect("failed to build ptx");
    let _ = bindings.write(PTX_STRING_PATH);
}

fn build_c_libs() {
    println!("cargo:rerun-if-changed=lib");
    // TODO: glob libs
    let dst = Config::new("lib/foo").build();

    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=static=foo");

    Config::new("lib/cuda_examples")
        .configure_arg(
            String::from("-DCMAKE_CUDA_ARCHITECTURES=")
                + &env_manager::cuda_compute_cap().to_string(),
        )
        .no_build_target(true)
        .build();
}

fn main() -> Result<(), Box<dyn Error>> {
    build_cuda_ptx();

    build_c_libs();

    Ok(())
}

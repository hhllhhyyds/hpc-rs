use std::{
    error::Error,
    path::{Path, PathBuf},
};

#[path = "src/cuda_env.rs"]
mod cuda_env;

fn build_ptx() {
    let builder = bindgen_cuda::Builder::default().kernel_paths_glob("src/cu/for_ptx/*.cu");
    let bindings = builder.build_ptx().unwrap();
    let _ = bindings.write("src/ptx.rs");
}

fn build_exe() {
    let nvcc_path = cuda_env::cuda_include_dir()
        .unwrap()
        .join("bin")
        .join("nvcc.exe");
    let src_glob: Vec<PathBuf> = glob::glob("src/cu/for_exe/*.cu")
        .expect("Invalid blob")
        .map(|p| p.expect("Invalid path"))
        .collect();
    let executable_name = src_glob
        .iter()
        .map(|p| {
            p.display()
                .to_string()
                .split_off("src/cu/for_exe/".len())
                .split('.')
                .next()
                .unwrap()
                .to_string()
        })
        .collect::<Vec<_>>();
    let src_path = src_glob
        .iter()
        .map(|p| PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap()).join(p));

    for (src, exe) in src_path.zip(executable_name) {
        let cap_str = format!("-arch=sm_{}", cuda_env::compute_cap());
        let out = Path::new(&std::env::var("OUT_DIR").unwrap()).join(&exe);
        let args = vec![
            "/C",
            nvcc_path.to_str().unwrap(),
            &cap_str,
            "-rdc=true",
            src.to_str().unwrap(),
            "-o",
            out.to_str().unwrap(),
            "-lcudadevrt",
        ];
        cuda_env::run_cmd(&args);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    build_ptx();
    build_exe();

    Ok(())
}

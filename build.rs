use std::{
    env,
    error::Error,
    path::{Path, PathBuf},
};

const CUDA_PTX_SRC_PATH_PATTERN: &str = "src/cu/for_ptx/*.cu";
const PTX_STRING_LOCATION: &str = "src/ptx.rs";
const CUDA_EXE_SRC_DIR: &str = "src/cu/for_exe";

fn build_ptx() {
    let builder = bindgen_cuda::Builder::default().kernel_paths_glob(CUDA_PTX_SRC_PATH_PATTERN);
    let bindings = builder.build_ptx().expect("failed to build ptx");
    let _ = bindings.write(PTX_STRING_LOCATION);
}

fn build_exe() {
    let nvcc_path = env_manager::nvcc_path().expect("failed to get nvcc path");

    let exe_names = {
        env::set_current_dir(env_manager::manifest_dir().join(Path::new(CUDA_EXE_SRC_DIR)))
            .expect("failed to set current dir");
        let exe_names: Vec<String> = glob::glob("*.cu")
            .expect("Invalid blob")
            .map(|p| {
                p.expect("Invalid path")
                    .display()
                    .to_string()
                    .split('.')
                    .next()
                    .expect("executable name wrong")
                    .to_string()
            })
            .collect();
        env::set_current_dir(env_manager::manifest_dir()).expect("failed to set current dir");
        exe_names
    };

    let src_pathes: Vec<PathBuf> = exe_names
        .iter()
        .map(|name| {
            env_manager::manifest_dir()
                .join(CUDA_EXE_SRC_DIR)
                .join(name.clone() + ".cu")
        })
        .collect();

    for (src, exe) in src_pathes.iter().zip(exe_names.iter()) {
        let cap_str = format!("-arch=sm_{}", env_manager::cuda_compute_cap());
        let out_path = Path::new(&std::env::var("OUT_DIR").expect("out dir wrong"))
            .join(exe.clone() + env_manager::EXE_SUFFIX);
        let args = vec![
            nvcc_path.to_str().unwrap(),
            &cap_str,
            "-rdc=true",
            src.to_str().unwrap(),
            "-o",
            out_path.to_str().unwrap(),
            "-lcudadevrt",
        ];
        env_manager::run_cmd(&args);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    build_ptx();
    build_exe();

    Ok(())
}

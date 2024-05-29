const FOREIGN_LIB_DIR_NAME: &str = "lib_gen";
const MY_CUDA_LIB_NAME: &str = "hpccuda";
const CUDA_LIB_SEARCH_PATH: &str = "/usr/local/cuda/lib64";

fn main() {
    println!("cargo::rerun-if-changed=src");
    println!("cargo::rerun-if-changed=benches");
    println!("cargo::rerun-if-changed=tests");

    let foreign_lib_dir = std::path::Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("target")
        .join(FOREIGN_LIB_DIR_NAME);
    if !std::path::Path::exists(&foreign_lib_dir) {
        std::fs::create_dir(&foreign_lib_dir).unwrap();
    }

    let builder = bindgen_cuda::Builder::default().out_dir(&foreign_lib_dir);
    builder.build_lib(foreign_lib_dir.join(format!("lib{}.a", MY_CUDA_LIB_NAME)));

    println!("cargo:rustc-link-search={}", foreign_lib_dir.display());
    println!("cargo:rustc-link-search={}", CUDA_LIB_SEARCH_PATH);

    println!("cargo:rustc-link-lib=static={}", MY_CUDA_LIB_NAME);
    println!("cargo:rustc-link-lib=static=cudart_static");
}

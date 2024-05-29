fn main() {
    println!("cargo::rerun-if-changed=src");
    println!("cargo::rerun-if-changed=benches");
    println!("cargo::rerun-if-changed=tests");

    let lib_dir_name = "lib_gen";
    let lib_directory =
        std::path::Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).join(lib_dir_name);
    if !std::path::Path::exists(&lib_directory) {
        std::fs::create_dir(&lib_directory).unwrap();
    }

    let builder = bindgen_cuda::Builder::default().out_dir(&lib_directory);

    let cuda_lib_name = "hpccuda";
    builder.build_lib(lib_directory.join(format!("lib{}.a", cuda_lib_name)));

    println!("cargo:rustc-link-search={}", lib_directory.display());
    println!("cargo:rustc-link-search=/usr/local/cuda/lib64");

    println!("cargo:rustc-link-lib=static={}", cuda_lib_name);
    println!("cargo:rustc-link-lib=cudart");

    // println!("cargo:rustc-link-lib=stdc++"); // .cu 其实是C++, NVCC会调用g++进行编译，所以需要C++标准库
    // println!("cargo:rustc-link-lib=cublas"); // 这是为了测试 ndarray-linalg 的 dot 函数
}

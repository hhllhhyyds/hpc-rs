pub mod cuda;
pub mod error;

pub fn cmake_outdir() -> std::path::PathBuf {
    let build_type = std::env::var("CMAKE_BUILD_TYPE")
        .unwrap_or("debug".to_string())
        .to_lowercase();
    let build_type = build_type[0..1].to_uppercase() + &build_type[1..];
    std::path::Path::new(env!("OUT_DIR"))
        .join("build")
        .join(build_type)
}

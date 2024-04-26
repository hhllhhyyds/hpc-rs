pub mod cuda;
pub mod error;

pub fn cmake_outdir() -> std::path::PathBuf {
    let out_dir = env!("OUT_DIR");
    let build_type = if out_dir.contains("debug") {
        "debug"
    } else if out_dir.contains("release") {
        "release"
    } else {
        panic!("OUT_DIR wrong");
    };

    let build_type = build_type[0..1].to_uppercase() + &build_type[1..];
    std::path::Path::new(out_dir).join("build").join(build_type)
}

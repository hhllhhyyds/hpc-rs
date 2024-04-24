use std::path::{Path, PathBuf};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(args[1] == "example");

    let example_exe = args[2].clone() + env_manager::EXE_SUFFIX;

    let mut example_path = env_manager::manifest_dir();
    example_path.push("target");
    example_path.push("debug");
    example_path.push("examples");
    example_path.push(example_exe);

    println!("example_path = {}", example_path.display());
    assert!(std::path::Path::exists(&example_path));

    // nsys_profile_help();
    profile(example_path, &args[3..]);
}

fn profile<T: AsRef<str>, P: AsRef<Path>>(program: P, programs_args: &[T]) {
    let nsys = nsys_path();

    let mut args = vec![
        nsys.to_str().unwrap(),
        "profile",
        "-o",
        "target/cuda-profile",
        "--stats=true",
        "--trace=cuda",
        "--force-overwrite=true",
        program.as_ref().to_str().unwrap(),
    ];
    args.extend(programs_args.iter().map(|x| x.as_ref()));

    let success = env_manager::run_cmd(&args);
    assert!(success)
}

#[allow(unused)]
fn nsys_profile_help() {
    let nsys = nsys_path();
    let args = vec![nsys.to_str().unwrap(), "profile", "--help"];
    let success = env_manager::run_cmd(&args);
    assert!(success)
}

fn nsys_path() -> PathBuf {
    let path = PathBuf::from(
        "C:/Program Files/NVIDIA Corporation/Nsight Systems 2023.4.4/target-windows-x64/nsys.exe",
    );
    assert!(path.is_file(), "nsys not found");
    path
}

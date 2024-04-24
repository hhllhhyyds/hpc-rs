use std::{
    io::{self, Write},
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(args[1] == "example");

    let example_exe = args[2].clone() + ".exe";

    let mut example_path: PathBuf = std::env::var_os("CARGO_MANIFEST_DIR").unwrap().into();
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
        "/C",
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

    run_cmd(&args);
}

#[allow(unused)]
fn nsys_profile_help() {
    let nsys = nsys_path();
    let args = vec!["/C", nsys.to_str().unwrap(), "profile", "--help"];
    run_cmd(&args);
}

fn nsys_path() -> PathBuf {
    let nsys_path_pattern = Path::new("C:\\")
        .join("Program Files")
        .join("NVIDIA Corporation")
        .join("Nsight Systems 2023.4.4")
        .join("target-windows-x64")
        .join("*nsys*");
    glob::glob(nsys_path_pattern.to_str().unwrap())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
}

fn run_cmd(args: &[&str]) {
    let mut cmd = Command::new("cmd");
    cmd.args(args);
    println!("cmd = {:?}", cmd);

    let output = cmd.output().expect("failed to execute process");

    println!("status: {}", output.status);
    io::stdout().write_all(&output.stdout).unwrap();
    io::stderr().write_all(&output.stderr).unwrap();

    assert!(output.status.success());
}

use std::env::consts::EXE_SUFFIX;

use crate::error::Error;

fn out_dir_build_type<'a>() -> (&'a str, &'a str) {
    let out_dir = env!("OUT_DIR");
    let build_type = if out_dir.contains("debug") {
        "debug"
    } else if out_dir.contains("release") {
        "release"
    } else {
        panic!("OUT_DIR wrong");
    };
    (out_dir, build_type)
}

pub fn cmake_outdir() -> std::path::PathBuf {
    let (out_dir, build_type) = out_dir_build_type();
    let build_type = build_type[0..1].to_uppercase() + &build_type[1..];
    std::path::Path::new(out_dir).join("build").join(build_type)
}

pub fn example_build_dir() -> std::path::PathBuf {
    let (_, build_type) = out_dir_build_type();
    env_manager::manifest_dir()
        .join("target")
        .join(build_type)
        .join("examples")
}

pub fn run_isolate_c_program(example: &str, args: &[&str]) -> Result<(), Error> {
    let exe = cmake_outdir().join(example.to_string() + EXE_SUFFIX);
    assert!(exe.is_file(), "{} not exists", exe.display());
    let cmd = [exe.to_str().unwrap()]
        .iter()
        .chain(args.iter())
        .map(|x| x.to_string())
        .collect::<Vec<_>>();
    let cmd_ref = cmd.iter().map(|x| x.as_ref()).collect::<Vec<&str>>();
    env_manager::run_cmd(&cmd_ref);

    Ok(())
}

pub fn run_isolate_c_program_with_args(example: &str) -> Result<(), Error> {
    let args = std::env::args().collect::<Vec<_>>();
    run_isolate_c_program(
        example,
        &args[1..].iter().map(AsRef::as_ref).collect::<Vec<_>>(),
    )
}

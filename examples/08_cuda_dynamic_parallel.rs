use std::env::consts::EXE_SUFFIX;

use hpc_rs::error::Error;

fn main() -> Result<(), Error> {
    let exe = hpc_rs::cmake_outdir().join("dynamic_parallel".to_string() + EXE_SUFFIX);
    env_manager::run_cmd(&[exe.to_str().unwrap()]);

    Ok(())
}

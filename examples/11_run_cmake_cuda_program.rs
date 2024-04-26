use std::env::consts::EXE_SUFFIX;

use hpc_rs::error::Error;

fn main() -> Result<(), Error> {
    let exe = hpc_rs::cmake_outdir().join("run_cuda_test".to_string() + EXE_SUFFIX);
    assert!(exe.is_file());
    env_manager::run_cmd(&[exe.to_str().unwrap()]);
    Ok(())
}

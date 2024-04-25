use std::path::Path;

use hpc_rs::error::Error;

fn main() -> Result<(), Error> {
    let exe = Path::new(env!("OUT_DIR"))
        .join("build")
        .join("Debug")
        .join("run_cuda_test.exe");
    env_manager::run_cmd(&[exe.to_str().unwrap()]);
    Ok(())
}

use std::path::Path;

use hpc_rs::error::Error;

fn main() -> Result<(), Error> {
    let exe = Path::new(env!("OUT_DIR")).join("dynamic_parallel.exe");
    env_manager::run_cmd(&[exe.to_str().unwrap()]);
    Ok(())
}
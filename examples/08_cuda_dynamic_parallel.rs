use std::path::Path;

use hpc_rs::error::Error;

fn main() -> Result<(), Error> {
    let exe = Path::new(env!("OUT_DIR")).join("dynamic_parallel.exe");
    let args = vec!["/C", exe.to_str().unwrap()];
    hpc_rs::cuda_env::run_cmd(&args);
    Ok(())
}

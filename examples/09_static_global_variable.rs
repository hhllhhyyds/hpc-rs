use hpc_rs::clib::run_isolate_c_program;
use hpc_rs::error::Error;

fn main() -> Result<(), Error> {
    run_isolate_c_program("static_variable", &[])
}

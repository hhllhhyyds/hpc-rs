use hpc_rs::clib::run_isolate_c_program_with_args;
use hpc_rs::error::Error;

fn main() -> Result<(), Error> {
    run_isolate_c_program_with_args("pinned_memory")
}

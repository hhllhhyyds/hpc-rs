use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let builder = bindgen_cuda::Builder::default().kernel_paths_glob("src/cu/static/*.cu");
    let bindings = builder.build_ptx().unwrap();
    let _ = bindings.write("src/ptx.rs");

    Ok(())
}

use std::{env, path::PathBuf};

#[test]
#[ignore = "manual"]
fn run_all_examples() {
    let manifest_dir: PathBuf = std::env::var_os("CARGO_MANIFEST_DIR").unwrap().into();
    let example_dir = manifest_dir.join("examples");
    env::set_current_dir(example_dir).unwrap();
    let examples: Vec<String> = glob::glob("*")
        .unwrap()
        .map(|p| {
            p.unwrap()
                .to_str()
                .unwrap()
                .to_string()
                .split(".")
                .next()
                .unwrap()
                .to_string()
        })
        .collect();
    env::set_current_dir(manifest_dir).unwrap();
    for example in examples.iter() {
        let success = hpc_rs::cuda_env::run_cmd(&["/C", "cargo", "run", "--example", &example]);
        assert!(success);
    }
}

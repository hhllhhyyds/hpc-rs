use std::{env, path::PathBuf};

#[test]
#[ignore = "manual"]
fn run_all_examples() {
    let manifest_dir: PathBuf = env_manager::manifest_dir();
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
        env_manager::run_cmd(&["cargo", "run", "--example", &example]);
    }
}

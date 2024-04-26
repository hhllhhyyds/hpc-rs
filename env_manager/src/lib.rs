use std::{env::consts::EXE_SUFFIX, path::PathBuf};

pub fn cuda_include_dir() -> Option<PathBuf> {
    let env_vars = [
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
        "CUDNN_LIB",
    ];

    let env_vars = env_vars
        .into_iter()
        .map(std::env::var)
        .filter_map(Result::ok)
        .map(Into::<PathBuf>::into);

    let roots = [
        "/usr",
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/lib/cuda",
        "C:/Program Files/NVIDIA GPU Computing Toolkit",
        "C:/CUDA",
    ];

    let roots = roots.into_iter().map(Into::<PathBuf>::into);

    env_vars
        .chain(roots)
        .find(|path| path.join("include").join("cuda.h").is_file())
}

pub fn nvcc_path() -> Option<PathBuf> {
    let exe = String::from("nvcc") + EXE_SUFFIX;
    let cuda_path = cuda_include_dir()?;
    let path = cuda_path.join("bin").join(exe);
    if path.is_file() {
        Some(path)
    } else {
        None
    }
}

pub fn cuda_compute_cap() -> usize {
    // Try to parse compute caps from env
    let compute_cap = if let Ok(compute_cap_str) = std::env::var("CUDA_COMPUTE_CAP") {
        compute_cap_str
            .parse::<usize>()
            .expect("Could not parse code")
    } else {
        // Use nvidia-smi to get the current compute cap
        let out = std::process::Command::new("nvidia-smi")
                .arg("--query-gpu=compute_cap")
                .arg("--format=csv")
                .output()
                .expect("`nvidia-smi` failed. Ensure that you have CUDA installed and that `nvidia-smi` is in your PATH.");
        let out = std::str::from_utf8(&out.stdout).expect("stdout is not a utf8 string");
        let mut lines = out.lines();
        assert_eq!(lines.next().expect("missing line in stdout"), "compute_cap");
        let cap = lines
            .next()
            .expect("missing line in stdout")
            .replace('.', "");
        let cap = cap.parse::<usize>().expect("cannot parse as int {cap}");
        println!("cargo:rustc-env=CUDA_COMPUTE_CAP={cap}");
        cap
    };

    // Grab available GPU codes from nvcc and select the highest one
    let (supported_nvcc_codes, max_nvcc_code) = {
        let out = std::process::Command::new("nvcc")
                .arg("--list-gpu-code")
                .output()
                .expect("`nvcc` failed. Ensure that you have CUDA installed and that `nvcc` is in your PATH.");
        let out = std::str::from_utf8(&out.stdout).expect("valid utf-8 nvcc output");

        let out = out.lines().collect::<Vec<&str>>();
        let mut codes = Vec::with_capacity(out.len());
        for code in out {
            let code = code.split('_').collect::<Vec<&str>>();
            if !code.is_empty() && code.contains(&"sm") {
                if let Ok(num) = code[1].parse::<usize>() {
                    codes.push(num);
                }
            }
        }
        codes.sort();
        let max_nvcc_code = *codes.last().expect("no gpu codes parsed from nvcc");
        (codes, max_nvcc_code)
    };

    // Check that nvcc supports the asked compute caps
    if !supported_nvcc_codes.contains(&compute_cap) {
        panic!(
            "nvcc cannot target gpu arch {compute_cap}. Available nvcc targets are {supported_nvcc_codes:?}."
        );
    }
    if compute_cap > max_nvcc_code {
        panic!(
            "CUDA compute cap {compute_cap} is higher than the highest gpu code from nvcc {max_nvcc_code}"
        );
    }

    compute_cap
}

#[cfg(target_os = "windows")]
pub fn run_cmd(args: &[&str]) {
    use std::{
        io::{self, Write},
        process::Command,
    };

    let mut cmd = Command::new("cmd");
    cmd.arg("/C");
    cmd.args(args);

    println!("cmd = {:?}", cmd);
    let output = cmd.output().expect("failed to execute process");
    io::stdout()
        .write_all(String::from_utf8_lossy(&output.stdout).as_bytes())
        .unwrap();
    io::stdout().flush().unwrap();
    io::stderr()
        .write_all(String::from_utf8_lossy(&output.stderr).as_bytes())
        .unwrap();
    io::stderr().flush().unwrap();

    assert!(output.status.success());
}

pub fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR").unwrap().into()
}

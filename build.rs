use std::{
    env,
    path::{Path, PathBuf},
};

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    if env::var_os("CARGO_FEATURE_NVJPEG").is_none() {
        return;
    }

    let mut roots: Vec<PathBuf> = Vec::new();
    if let Some(root) = env::var_os("CUDA_HOME") {
        roots.push(root.into());
    }
    if let Some(root) = env::var_os("CUDA_PATH") {
        roots.push(root.into());
    }
    roots.push("/usr/local/cuda".into());
    roots.push("/usr/local/cuda-13.0".into());
    roots.push("/usr/local/cuda-12".into());

    for root in roots {
        for rel in ["targets/x86_64-linux/lib", "lib64", "lib"] {
            let candidate = Path::new(&root).join(rel);
            if candidate.join("libnvjpeg.so").exists() {
                println!("cargo:rustc-link-search=native={}", candidate.display());
                return;
            }
        }
    }
}

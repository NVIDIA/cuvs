/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

mod cmake;

#[cfg(feature = "generate-bindings")]
use std::path::{Path, PathBuf};

#[cfg(feature = "generate-bindings")]
fn generate_bindings(include_dir: &Path, include_dirs: &[PathBuf]) {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set by Cargo"));
    let stub_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("bindgen-stubs");

    let mut builder = bindgen::Builder::default()
        .header("cuvs_c_wrapper.h")
        .must_use_type("cuvsError_t")
        .allowlist_function("cuvs.*")
        .allowlist_type("(cuvs|DL).*")
        .rustified_enum("(cuvs|DL).*")
        .blocklist_item("cuda.*")
        .blocklist_item("CUstream_st")
        .raw_line("use crate::{cudaDataType_t, cudaStream_t};")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

    builder = builder.clang_arg(format!("-I{}", stub_dir.display()));
    builder = builder.clang_arg(format!("-I{}", include_dir.display()));

    for include_dir in include_dirs {
        builder = builder.clang_arg(format!("-I{}", include_dir.display()));
    }

    builder
        .generate()
        .expect("bindgen failed to generate cuvs bindings")
        .write_to_file(out_dir.join("cuvs_bindings.rs"))
        .expect("failed to write cuvs_bindings.rs");
}

fn main() {
    println!("cargo::rerun-if-changed=cmake.rs");
    println!("cargo::rerun-if-changed=bindgen-stubs/cuda_runtime.h");
    println!("cargo::rerun-if-env-changed=CMAKE_PREFIX_PATH");
    println!("cargo::rerun-if-env-changed=CONDA_PREFIX");
    println!("cargo::rerun-if-env-changed=LIBCUVS_USE_PYTHON");
    println!("cargo::rerun-if-env-changed=VIRTUAL_ENV");

    if cfg!(feature = "doc-only") {
        return;
    }

    let metadata = match cmake::locate_cuvs() {
        Ok(metadata) => metadata,
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
    };

    // Expose include path to downstream crates via DEP_CUVS_INCLUDE.
    println!("cargo::metadata=include={}", metadata.include_dir.display());
    // Expose the directory containing libcuvs_c.so via DEP_CUVS_LIB.
    println!("cargo::metadata=lib={}", metadata.lib_dir.display());

    #[cfg(feature = "generate-bindings")]
    generate_bindings(&metadata.include_dir, &metadata.bindgen_include_dirs);
}

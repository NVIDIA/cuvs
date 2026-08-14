/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::env;

fn add_runtime_search_path(var_name: &str) {
    if let Ok(lib_path) = env::var(var_name) {
        println!("cargo:rustc-link-arg=-Wl,-rpath={lib_path}");
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=DEP_CUVS_LIB");
    add_runtime_search_path("DEP_CUVS_LIB");
}

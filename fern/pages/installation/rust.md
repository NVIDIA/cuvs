# Rust Installation

Use this page when you need the NVIDIA cuVS Rust bindings. The Rust crate depends on the native NVIDIA cuVS C API, so `libcuvs.so` and `libcuvs_c.so` must be available when compiling and running Rust code.

All NVIDIA cuVS routine implementations live in the C++ core. The Rust bindings call into the native C and C++ libraries, so install both `libcuvs_c` and `libcuvs`.

## Install Pre-Compiled Dependencies

Install the native libraries first:

```bash
# CUDA 13
conda install -c rapidsai -c conda-forge libcuvs cuda-version=13.3

# CUDA 12
conda install -c rapidsai -c conda-forge libcuvs cuda-version=12.9
```

When building examples or working on the Rust bindings from source, use the centralized [build environment guidance](/installation#create-a-build-environment). A Rust-specific environment YAML file is also available in `conda/environments` when you want a narrower development environment.

Configure `LIBCLANG_PATH` so Rust bindgen can find libclang:

```bash
LIBCLANG_PATH=$(dirname "$(find "$CONDA_PREFIX" -name libclang.so | head -n 1)")
export LIBCLANG_PATH
```

Add NVIDIA cuVS to your Rust project:

```toml
[dependencies]
cuvs = ">=24.6.0"
```

Then build or run your project with Cargo:

```bash
cargo build
```

## Build From Source

Before building from source, review the [shared C++ source-build prerequisites](/installation#build-from-source), including the recommended conda environment setup for build dependencies.

Build the native libraries first if needed:

```bash
./build.sh libcuvs
```

Then build the Rust bindings:

```bash
./build.sh rust
```

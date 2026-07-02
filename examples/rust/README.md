# cuVS Rust Example

This template project provides a drop-in sample to for using cuVS in a rust project.

First, please refer to our [installation docs](https://docs.rapids.ai/api/cuvs/stable/build.html#cuda-gpu-requirements) for the minimum requirements to use cuVS. Note that you will have to have the libcuvs.so and libcuvs_c.so binaries installed to compile this example project.

Once the minimum requirements are satisfied, this example template application can be built using 'cargo build', and this directory can be copied directly in order to build a new application with cuVS.

You may follow these steps to quickly get set up:

```bash
conda env create --name rust -f conda/environments/rust_cuda-133_arch-$(uname -m).yaml
conda activate rust
```
You may prefer to use `mamba`, as it provides significant speedup over `conda`.

1. Set up the required environment variables:
```bash
LIBCLANG_PATH=$(dirname "$(find "$CONDA_PREFIX" -name libclang.so | head -n 1)")
export LIBCLANG_PATH
echo "LIBCLANG_PATH=$LIBCLANG_PATH"
```

2. Add the cuvs dependency in your project:
```TOML
# Cargo.toml
[dependencies]
cuvs = ">=24.6.0"
#...rest of your dependencies
```
Then you can run your project with `cargo run`.

See [main.rs](./src/main.rs) for an example implementation.

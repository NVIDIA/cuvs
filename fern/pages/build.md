# Installation

NVIDIA cuVS provides APIs for C, C++, Python, Java, Go, and Rust. Start with the language you plan to use; each guide separates package installation from source builds and calls out any language-specific setup.

All NVIDIA cuVS routine implementations live in the C++ core. For every non-C++ language binding, install both the C library (`libcuvs_c`) and the C++ library (`libcuvs`) unless the selected package explicitly bundles them.

## CUDA GPU Requirements

Pre-compiled NVIDIA cuVS packages are available for Linux on x86_64 and aarch64. Native Windows support is not available at this time. On Windows, use WSL2 with GPU passthrough. See the [RAPIDS WSL2 guide](https://rapids.ai/start.html#wsl2).

Source builds and package installs require a supported NVIDIA GPU. For current source builds, use CUDA Toolkit 12.2 or newer and an Ampere architecture GPU or newer, which means compute capability 8.0 or higher.

## Language Guides

- [C](/installation/c): install or build the C API and `libcuvs_c`.
- [C++](/installation/cpp): install or build the C++ headers and `libcuvs`.
- [Python](/installation/python): install Python wheels or conda packages, or build the Python package from source.
- [Java](/installation/java): build the Java API and connect it to matching native NVIDIA cuVS libraries.
- [Go](/installation/go): install the Go module and configure CGO against native NVIDIA cuVS libraries.
- [Rust](/installation/rust): install the Rust crate and configure native NVIDIA cuVS dependencies.

## Build From Source

Most source builds use the repository `build.sh` script. The script wraps CMake, prepares install targets, and provides language-specific build targets. Each language guide shows the target most users need.

The common source-build prerequisites are:

1. CMake 3.26.4 or newer.
2. GCC 9.3 or newer, with GCC 11.4 or newer recommended.
3. CUDA Toolkit 12.2 or newer.
4. An Ampere architecture GPU or newer.

### Create a Build Environment

The recommended way to construct an environment with the dependencies required to build NVIDIA cuVS is to use conda with the repository environment YAML file:

```bash
conda env create --name cuvs -f conda/environments/all_cuda-133_arch-$(uname -m).yaml
conda activate cuvs
```

You may prefer `mamba` over `conda` for faster environment solves. The `conda/environments` directory also contains language-specific environment YAML files for narrower development environments. Conda is not required, but if you do not use it, install all required build dependencies explicitly before running `build.sh`.

## Documentation Preview

The NVIDIA cuVS documentation is a Fern project in the repository's `fern` directory. Fern requires Node.js 22 or newer. If the docs fail with an error such as `SyntaxError: Unexpected token '.'`, check `node --version` and activate a newer Node.js runtime.

Run the local preview from the repository root:

```bash
fern/build_docs.sh dev
```

Fern serves the preview at [http://localhost:3000](http://localhost:3000) by default.

Run the Fern checks before publishing documentation changes:

```bash
fern/build_docs.sh check
```

# C Installation

Use this page when you need the NVIDIA cuVS C API, headers, and `libcuvs_c` shared library. The C API is the stable ABI boundary used by downstream integrations and language bindings.

All NVIDIA cuVS routine implementations live in the C++ core. The C API calls into that core, so C applications need both `libcuvs_c` and `libcuvs` installed at runtime.

## Install Pre-Compiled Packages

The easiest way to install the C API is through conda. Use [miniforge](https://github.com/conda-forge/miniforge) for a minimal conda installation, and prefer `mamba` when available.

```bash
# CUDA 13
conda install -c rapidsai -c conda-forge libcuvs cuda-version=13.3

# CUDA 12
conda install -c rapidsai -c conda-forge libcuvs cuda-version=12.9
```

The `libcuvs` package installs the C headers, C++ headers, `libcuvs_c`, and `libcuvs`.

### Tarball

Pre-built tarballs are available from [developer.nvidia.com/cuvs-downloads](https://developer.nvidia.com/cuvs-downloads). Tarball installs require NCCL, `libopenmp`, CUDA Toolkit runtime 12.2 or newer, and an Ampere architecture GPU or newer.

Download the tarball for your CPU architecture and CUDA version, then extract it:

```bash
tar -xzvf libcuvs-linux-sbsa-26.02.00.189485_cuda12-archive.tar.xz -C /path/to/folder
```

Add the extracted library directory to your loader path:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/folder
```

## Build From Source

Before building from source, review the [shared C++ source-build prerequisites](/installation#build-from-source), including the recommended conda environment setup for build dependencies.

Build and install the native C and C++ libraries together:

```bash
./build.sh libcuvs
```

This installs `libcuvs_c.so`, `libcuvs.so`, headers, and downloaded dependencies into `$INSTALL_PREFIX` by default. Pass `-n` to build without installing.

Uninstall the native libraries with:

```bash
./build.sh libcuvs --uninstall
```

Disable multi-GPU features with:

```bash
./build.sh libcuvs --no-mg
```

Build the C and C++ tests with:

```bash
./build.sh libcuvs tests
```

Build a limited set of tests with:

```bash
./build.sh libcuvs tests -n --limit-tests='NEIGHBORS_TEST;CAGRA_C_TEST'
```

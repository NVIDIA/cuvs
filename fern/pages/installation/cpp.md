# C++ Installation

Use this page when you need the NVIDIA cuVS C++ headers, `libcuvs` shared library, and native C++ APIs.

All NVIDIA cuVS routine implementations live in the C++ core. The C++ API links directly to `libcuvs`; the C API and all non-C++ language bindings also need `libcuvs_c` installed with `libcuvs`.

## Install Pre-Compiled Packages

The easiest way to install the C++ API is through conda. Use [miniforge](https://github.com/conda-forge/miniforge) for a minimal conda installation, and prefer `mamba` when available.

```bash
# CUDA 13
conda install -c rapidsai -c conda-forge libcuvs cuda-version=13.3

# CUDA 12
conda install -c rapidsai -c conda-forge libcuvs cuda-version=12.9
```

The `libcuvs` package installs the C++ headers, C headers, `libcuvs`, and `libcuvs_c`.

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

This installs `libcuvs.so`, `libcuvs_c.so`, headers, and downloaded dependencies into `$INSTALL_PREFIX` by default. Pass `-n` to build without installing.

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

## Use CMake Directly

Use the root `build.sh` script for most builds. When you need finer CMake control, configure from the `cpp` directory:

```bash
cd cpp
mkdir build
cd build
cmake -D BUILD_TESTS=ON -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ../
make -j<parallel_level> install
```

Common CMake flags include:

| Flag | Possible Values | Default Value | Behavior |
| --- | --- | --- | --- |
| `BUILD_TESTS` | `ON`, `OFF` | `ON` | Compile Googletests |
| `CUDA_ENABLE_KERNELINFO` | `ON`, `OFF` | `OFF` | Enable `kernelinfo` in nvcc for `compute-sanitizer` |
| `CUDA_ENABLE_LINEINFO` | `ON`, `OFF` | `OFF` | Enable the `-lineinfo` option for nvcc |
| `CUDA_STATIC_MATH_LIBRARIES` | `ON`, `OFF` | `OFF` | Statically link the CUDA math libraries |
| `DETECT_CONDA_ENV` | `ON`, `OFF` | `ON` | Enable conda environment detection for dependencies |
| `CUVS_NVTX` | `ON`, `OFF` | `OFF` | Enable NVTX markers |

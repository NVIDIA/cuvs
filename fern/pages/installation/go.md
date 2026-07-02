# Go Installation

Use this page when you need the NVIDIA cuVS Go bindings. The Go bindings use CGO and require the native NVIDIA cuVS C and C++ libraries.

All NVIDIA cuVS routine implementations live in the C++ core. The Go bindings call into the native C and C++ libraries, so install both `libcuvs_c` and `libcuvs`.

## Install Pre-Compiled Dependencies

Install the native libraries first:

```bash
# CUDA 13
conda install -c rapidsai -c conda-forge libcuvs cuda-version=13.3

# CUDA 12
conda install -c rapidsai -c conda-forge libcuvs cuda-version=12.9
```

When building examples or working on the Go bindings from source, use the centralized [build environment guidance](/installation#create-a-build-environment). A Go-specific environment YAML file is also available in `conda/environments` when you want a narrower development environment.

Configure CGO so Go can find the native headers and libraries:

```bash
export CUDA_HOME="/usr/local/cuda"
export CGO_CFLAGS="-I${CONDA_PREFIX}/include -I${CUDA_HOME}/include"
export CGO_LDFLAGS="-L${CONDA_PREFIX}/lib -lcuvs -lcuvs_c"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export CC=clang
```

Add the Go module to your project:

```bash
go get github.com/rapidsai/cuvs/go@v26.06.00
```

Then build your project with the usual Go tooling:

```bash
go build ./...
```

## Build From Source

Before building from source, review the [shared C++ source-build prerequisites](/installation#build-from-source), including the recommended conda environment setup for build dependencies.

Build the native libraries first if needed:

```bash
./build.sh libcuvs
```

Then build the Go bindings:

```bash
./build.sh go
```

The Go bindings require CGO, so builds will fail if the native libraries are not installed or if the CGO include and library paths are not configured correctly.

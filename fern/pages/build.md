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

## Build the Standalone C Library with Docker

Use the standalone Docker build when you want a `libcuvs_c.tar.gz` archive that you can unpack and use to build your own C or C++ binaries for deployment or integration.

### Prerequisites

- Docker with support for the target platform: x86_64 or aarch64.
- At least 16 GB of memory and 20 GB of free disk space available to Docker.
- NVIDIA Container Toolkit and a GPU if you want to run GPU-dependent steps. The image is based on CUDA and may require GPU support at runtime.

### Use the Helper Script

From the repository root, run:

```bash
./build.sh tarball
```

The script builds the Docker image, runs the build in a container, writes the tarball to `./build/libcuvs_c.tar.gz`, and copies it to `./libcuvs_c.tar.gz` for CI artifact upload and convenience.

To select CUDA and Python versions, set environment variables to exact versions from a valid [`rapidsai/ci-wheel` image tag](https://hub.docker.com/r/rapidsai/ci-wheel/tags). For example:

```bash
CUDA_VERSION=12.9.2 PYTHON_VERSION=3.11 ./build.sh tarball
```

To write the tarball to another directory, set `BUILD_OUTPUT_DIR`:

```bash
BUILD_OUTPUT_DIR=$(pwd)/dist ./build.sh tarball
```

The tarball is written to `/path/to/output/libcuvs_c.tar.gz` and is also copied to the repository root.

To build and install the C library tests in the archive, pass `--tarball-build-tests`:

```bash
./build.sh tarball --tarball-build-tests
```

### Tarball Contents

The archive contains the headers, libraries, CMake configuration, and license information needed to compile and link C or C++ applications against the standalone NVIDIA cuVS libraries.

CI checks out the [RAPIDS SPDX license builder](https://github.com/rapidsai/spdx-license-builder) before running the container so that the generated license artifacts are included in the tarball.

### Build and Run the Docker Image Manually

If you do not want to use the helper script, build the image directly from the repository root:

```bash
docker build -f Dockerfile.standalone -t cuvs-standalone-c .
```

This command builds from the published `rapidsai/ci-wheel` base image and tags the resulting local image as `cuvs-standalone-c`. The following `docker run` examples use that same local tag; Docker does not pull an image named `cuvs-standalone-c` from a registry.

Run the build and mount the repository plus an output directory:

```bash
mkdir -p build
docker run --rm \
  -v "$(pwd):/workspace" \
  -v "$(pwd)/build:/build" \
  cuvs-standalone-c
```

To select different CUDA and Python versions, pass build arguments:

```bash
docker build -f Dockerfile.standalone \
  --build-arg CUDA_VERSION=12.9.2 \
  --build-arg PYTHON_VERSION=3.11 \
  -t cuvs-standalone-c .
```

Mount another host directory at `/build` to change the output location:

```bash
mkdir -p /path/to/output
docker run --rm \
  -v "$(pwd):/workspace" \
  -v "/path/to/output:/build" \
  cuvs-standalone-c
```

Pass `--tarball-build-tests` to include the C library tests:

```bash
mkdir -p build
docker run --rm \
  -v "$(pwd):/workspace" \
  -v "$(pwd)/build:/build" \
  cuvs-standalone-c --tarball-build-tests
```

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

# Building the standalone C library with Docker

This guide is for anyone who wants to build the cuVS standalone C library (`libcuvs_c`) and get a single tarball they can unpack and use to build their own binaries (e.g. for deployment or integration).

## Prerequisites

- **Docker** (with support for the platform you’re building for: x86_64 or aarch64).
- **Enough resources**: the build is heavy; ensure Docker has sufficient memory and disk (e.g. 16GB+ RAM, 20GB+ free disk).
- **NVIDIA Container Toolkit** (and a GPU) if you want to run GPU-dependent steps; the image is based on CUDA and may require GPU support in the runtime.

## Quick start: use the helper script

From the **root of the cuVS repo**, run:

```bash
ci/run_standalone_c_docker.sh
```

The script builds the Docker image (if needed), runs the build inside a container, and writes the tarball to **`./build/libcuvs_c.tar.gz`**. It also copies it to **`./libcuvs_c.tar.gz`** in the repo root for convenience.

### Custom CUDA or Python version

Set environment variables before running the script. Use values that match a valid [rapidsai/ci-wheel](https://hub.docker.com/r/rapidsai/ci-wheel/tags) tag (e.g. `26.04-cuda12.4-rockylinux8-py3.10`):

```bash
CUDA_VERSION=12.4 PYTHON_VERSION=3.10 ci/run_standalone_c_docker.sh
```

### Different output directory

```bash
BUILD_OUTPUT_DIR=/path/to/your/output ci/run_standalone_c_docker.sh
```

The tarball will be at `/path/to/your/output/libcuvs_c.tar.gz`; the script still copies it to the repo root as `libcuvs_c.tar.gz` unless you change the script.

### Include C library tests

To also build and install the C library tests (for local testing):

```bash
ci/run_standalone_c_docker.sh --build-tests
```

## What’s in the tarball

After extracting `libcuvs_c.tar.gz` you get a layout suitable for building and linking your own C/C++ code: headers, static and/or shared libraries, CMake config (if present), and license info. Use the included headers and link against the provided libraries in your build system.

## License builder (for CI)

CI checks out the [RAPIDS license builder](https://github.com/rapidsai/spdx-license-builder) into `./tool/` before running the container so the image can generate the license artifacts that go into the tarball. For a local build you don’t need to do that unless you want the same license output; the image will still produce a valid `libcuvs_c.tar.gz` without it.

## Docker build and run (optional)

If you prefer to build and run the image yourself instead of using the helper script:

```bash
# Build the image (defaults: CUDA 13.0, Python 3.11)
docker build -f ci/standalone_c/Dockerfile.standalone_c -t cuvs-standalone-c .

# Run the build; the tarball will appear in ./build on your machine
docker run --rm \
  -v "$(pwd):/workspace" \
  -v "$(pwd)/build:/build" \
  cuvs-standalone-c
```

**Custom CUDA or Python version:** add build args when building the image:

```bash
docker build -f ci/standalone_c/Dockerfile.standalone_c \
  --build-arg CUDA_VERSION=12.4 \
  --build-arg PYTHON_VERSION=3.10 \
  -t cuvs-standalone-c .
```

**Different output directory:** mount your chosen host path at `/build`:

```bash
docker run --rm \
  -v "$(pwd):/workspace" \
  -v "/path/to/your/output:/build" \
  cuvs-standalone-c
```

**Include C library tests:** pass `--build-tests` to the container:

```bash
docker run --rm \
  -v "$(pwd):/workspace" \
  -v "$(pwd)/build:/build" \
  cuvs-standalone-c --build-tests
```

# Docker Build Environment for cuVS Java API

This directory contains the Docker-based build system for the cuVS Java API, providing a containerized environment with all necessary dependencies for building the project across different CUDA versions and architectures.

## Overview

The Docker build system provides:
- Consistent build environment across different host systems
- Support for multiple CUDA versions (12.x and 13.x)
- Cross-platform builds (x86_64 and aarch64)
- Pre-configured development tools (GCC toolset, CMake, Maven, JDK 22)

## Quick Start

### Default Build (CUDA 12, Rocky Linux 9, All GPU Architectures)

```bash
./build-in-docker
```

This builds using the defaults:
- CUDA version: 12.9.1
- OS version: Rocky Linux 9
- Target: libcuvs java with all GPU architectures (`--allgpuarch`)

### Building for CUDA 13, All GPU Architectures

```bash
CUDA_VERSION=13.0.0 ./build-in-docker
```

### Building for Local GPU Architecture Only

```bash
./build-in-docker libcuvs java
```

## Environment Variables

### Core Configuration

- **`CUDA_VERSION`**: CUDA toolkit version to use (default: `12.9.1`)
  - Examples: `12.9.1`, `13.0.2`, `13.3.0`
- **`OS_VERSION`**: Rocky Linux OS version to use (default: `9`)
  - Supported values: `8`, `9`
- **`CMAKE_GENERATOR`**: CMake generator to use (default: `Ninja`)
- **`LOCAL_MAVEN_REPO`**: Local Maven repository path (default: `$HOME/.m2/repository`)

### Docker Configuration

- **`JNI_DOCKER_DEV_BUILD`**: Set to `ON` for development builds with gcc-toolset enabled by default (default: `OFF`)
- **`DOCKER_CMD`**: Docker command to use (default: `docker`)
- **`DOCKER_BUILD_EXTRA_ARGS`**: Additional arguments for `docker build`
- **`DOCKER_RUN_EXTRA_ARGS`**: Additional arguments for `docker run`
- **`DOCKER_GPU_OPTS`**: GPU options for Docker (default: `--gpus all`)

### Build Optimization

- **`LOCAL_CCACHE_DIR`**: ccache directory for build acceleration (default: `$HOME/.ccache`)
- **`PARALLEL_LEVEL`**: Number of parallel build jobs
- **`VERBOSE`**: Enable verbose build output

## Architecture Support

The build system automatically detects the host architecture:
- **x86_64**: Uses `linux/amd64` platform and x86_64 CMake binaries
- **aarch64**: Uses `linux/arm64` platform and aarch64 CMake binaries

## Files in This Directory

- **`build-in-docker`**: Main entry point script for Docker-based builds
- **`run-in-docker`**: Lower-level script that handles Docker container execution
- **`Dockerfile`**: Multi-stage Docker image definition with CUDA, development tools, and dependencies
- **`env.sh`**: Environment configuration script

## Examples

### Development Build with Custom CUDA Version (Default: All GPU Architectures)

```bash
CUDA_VERSION=13.3.0 JNI_DOCKER_DEV_BUILD=ON ./build-in-docker
```

### Production Build with Rocky Linux 8

```bash
OS_VERSION=8 ./build-in-docker
```

### Build with Custom Maven Repository and ccache

```bash
LOCAL_MAVEN_REPO=/custom/maven/repo LOCAL_CCACHE_DIR=/custom/ccache CUDA_VERSION=13.0.0 ./build-in-docker
```

### Build with Additional Docker Arguments

```bash
DOCKER_BUILD_EXTRA_ARGS="--no-cache" ./build-in-docker
```

### Interactive Development Session

```bash
CUDA_VERSION=12.9.1 JNI_DOCKER_DEV_BUILD=ON ./docker-build/run-in-docker
```

This starts an interactive bash shell in the container for development work.

## Docker Image Details

The Docker image is based on `nvidia/cuda:{CUDA_VERSION}-devel-rockylinux9` and includes:

- **CUDA Development Tools**: Complete CUDA toolkit for the specified version
- **GCC Toolset 14**: Modern C++ compiler with C++20 support
- **CMake 4.0**: Build system generator
- **JDK 22**: Java Development Kit with Panama FFM support
- **Maven**: Java project management and build tool
- **ccache**: Compiler cache for faster incremental builds
- **Additional Tools**: git, ninja-build, wget, tar, zip, patch

## Troubleshooting

### CUDA Version Compatibility

Ensure your target CUDA version is supported by checking available tags at:
- [NVIDIA CUDA Docker Hub](https://hub.docker.com/r/nvidia/cuda/tags)

### GPU Access Issues

If you encounter GPU access problems, verify:
1. NVIDIA Docker runtime is installed (`nvidia-docker2` or `nvidia-container-toolkit`)
2. Your user has permissions to access Docker
3. GPU drivers are compatible with the CUDA version

### Build Performance

For tweaking build performance:
1. Use ccache: Ensure `LOCAL_CCACHE_DIR` is set to a persistent directory. (By default, the build uses ccache.)
2. Change parallelism: Set `PARALLEL_LEVEL` to match your CPU cores. (This happens by default.)
3. Use development builds: Set `JNI_DOCKER_DEV_BUILD=ON` for development

### Memory Requirements

The Docker build process requires significant memory. For large projects, ensure:
- At least 8GB of available RAM
- Sufficient disk space for Docker images and build artifacts

## Related Documentation

- [cuVS Java API README](../README.md): Main Java API documentation
- [cuVS Build Instructions](https://docs.rapids.ai/api/cuvs/stable/build/): Native build documentation

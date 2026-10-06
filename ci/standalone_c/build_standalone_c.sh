#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build script for the standalone C library.
#
# Use 'Dockerfile.standalone' to build an image with all the prerequisites
# and run this in a container.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TOOLSET_VERSION=14

BUILD_C_LIB_TESTS="OFF"
if [[ "${1:-}" == "--tarball-build-tests" ]]; then
  BUILD_C_LIB_TESTS="ON"
fi

source rapids-install-sccache
source rapids-configure-sccache

PIP_PACKAGES=(
  'cmake>=4.0'
  'git+https://github.com/rapidsai/spdx-license-builder.git'
  'ninja>=1.13'
)

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
if [[ "${RAPIDS_CUDA_MAJOR}" == "13" ]]; then
  PIP_PACKAGES+=(
    cuda-tile
  )
fi

rapids-pip-retry install "${PIP_PACKAGES[@]}"
pyenv rehash

rapids-logger "Begin cpp build"

sccache --stop-server 2>/dev/null || true

scl enable gcc-toolset-${TOOLSET_VERSION} -- \
      cmake -S cpp -B cpp/build/ -GNinja \
            -DCMAKE_CUDA_HOST_COMPILER=/opt/rh/gcc-toolset-${TOOLSET_VERSION}/root/usr/bin/gcc \
            -DCMAKE_CUDA_ARCHITECTURES=RAPIDS \
            -DCUTLASS_ENABLE_TESTS=OFF \
            -DDISABLE_OPENMP=OFF \
            -DBUILD_TESTS=OFF \
            -DBUILD_SHARED_LIBS=ON \
            -DCUVS_STATIC_RAPIDS_LIBRARIES=ON
cmake --build cpp/build "-j${PARALLEL_LEVEL}"

sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true

rapids-logger "Begin c build"

scl enable gcc-toolset-${TOOLSET_VERSION} -- \
      cmake -S c -B c/build -GNinja \
            -DCMAKE_CUDA_HOST_COMPILER=/opt/rh/gcc-toolset-${TOOLSET_VERSION}/root/usr/bin/gcc \
            -DCUVSC_STATIC_CUVS_LIBRARY=ON \
            -DCMAKE_PREFIX_PATH="$PWD/cpp/build/" \
            -DBUILD_TESTS=${BUILD_C_LIB_TESTS}
cmake --build c/build "-j${PARALLEL_LEVEL}"

sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true

rapids-logger "Begin c install"
cmake --install c/build --prefix c/build/install

# libcuvs_c contains libcuvs_static, whose ACE implementation uses the private KvikIO shared
# library. Bundle that runtime library without adding KvikIO headers or CMake metadata to the
# standalone C artifact.
if ! cmake --install cpp/build \
      --prefix c/build/install \
      --component cuvs_standalone_runtime; then
  echo "Error: failed to install component 'cuvs_standalone_runtime' from cpp/build" >&2
  exit 1
fi

# need to install the tests
if [ "${BUILD_C_LIB_TESTS}" != "OFF" ]; then
      cmake --install c/build --prefix c/build/install --component testing
fi

rapids-logger "Begin gathering licenses"
license-builder . --output-json c/build/install/licenses.json --output-txt c/build/install/LICENSE

rapids-logger "Begin c tarball creation"
"${REPO_ROOT}/build.sh" tarball

#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

TOOLSET_VERSION=14
NINJA_VERSION=v1.13.1

BUILD_C_LIB_TESTS="OFF"
if [[ "${1:-}" == "--build-tests" ]]; then
  BUILD_C_LIB_TESTS="ON"
fi

dnf install -y \
      patch \
      tar \
      unzip \
      wget

if ! command -V ninja >/dev/null 2>&1; then
    case "$(uname -m)" in
        x86_64)
            wget --no-hsts -q -O /tmp/ninja-linux.zip "https://github.com/ninja-build/ninja/releases/download/${NINJA_VERSION}/ninja-linux.zip";
            ;;
        aarch64)
            wget --no-hsts -q -O /tmp/ninja-linux.zip "https://github.com/ninja-build/ninja/releases/download/${NINJA_VERSION}/ninja-linux-aarch64.zip";
            ;;
        *)
            echo "Unrecognized platform '$(uname -m)'" >&2
            exit 1
            ;;
    esac
    unzip -d /usr/bin /tmp/ninja-linux.zip
    chmod +x /usr/bin/ninja
    rm /tmp/ninja-linux.zip
fi

source rapids-install-sccache
source rapids-configure-sccache
source rapids-date-string

rapids-pip-retry install cmake
pyenv rehash

rapids-print-env

rapids-logger "Begin cpp build"

sccache --stop-server 2>/dev/null || true

RAPIDS_PACKAGE_VERSION=$(rapids-generate-version)
export RAPIDS_PACKAGE_VERSION

RAPIDS_ARTIFACTS_DIR=${RAPIDS_ARTIFACTS_DIR:-"${PWD}/artifacts"}
mkdir -p "${RAPIDS_ARTIFACTS_DIR}"
export RAPIDS_ARTIFACTS_DIR

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
cmake --install cpp/build \
      --prefix c/build/install \
      --component cuvs_standalone_runtime

# need to install the tests
if [ "${BUILD_C_LIB_TESTS}" != "OFF" ]; then
      cmake --install c/build --prefix c/build/install --component testing
fi


rapids-logger "Begin gathering licenses"
rapids-pip-retry install git+https://github.com/rapidsai/spdx-license-builder.git
license-builder . --output-json c/build/install/licenses.json --output-txt c/build/install/LICENSE

rapids-logger "Begin c tarball creation"
tar czf libcuvs_c.tar.gz -C c/build/install/ .
ls -lh libcuvs_c.tar.gz

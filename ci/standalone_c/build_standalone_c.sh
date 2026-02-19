#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# Build script for the standalone C library. Expects to be run inside an
# environment that already provides: dnf packages (patch, tar, unzip, wget),
# ninja, cmake (e.g. ci/standalone_c/Dockerfile.standalone_c).

set -euo pipefail

TOOLSET_VERSION=14

BUILD_C_LIB_TESTS="OFF"
if [[ "${1:-}" == "--build-tests" ]]; then
  BUILD_C_LIB_TESTS="ON"
fi

source rapids-install-sccache
source rapids-configure-sccache
source rapids-date-string

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

#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

rapids-pip-retry install cmake
pyenv rehash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_PREFIX="${PWD}/libcuvs_c_install"
mkdir -p "${INSTALL_PREFIX}"

# Download the standalone C library artifact
if [ -z "$1" ]; then
  echo "Error: name of the standalone C library artifact is missing"
  exit 1
fi

payload_name="$1"
pkg_name="libcuvs_c.tar.gz"
rapids-logger "Download ${payload_name} artifacts from previous jobs"
DOWNLOAD_LOCATION=$(rapids-download-from-github "${payload_name}")

# Extract the artifact to a staging directory
tar -xf "${DOWNLOAD_LOCATION}/${pkg_name}" -C "${INSTALL_PREFIX}"

rapids-logger "Validate C API shared library"
C_API_LIBRARY=""
for C_API_LIBRARY_DIR in "${INSTALL_PREFIX}/lib" "${INSTALL_PREFIX}/lib64"; do
  if [[ -f "${C_API_LIBRARY_DIR}/libcuvs_c.so" ]]; then
    C_API_LIBRARY="${C_API_LIBRARY_DIR}/libcuvs_c.so"
    break
  fi
done

if [[ -z "${C_API_LIBRARY}" ]]; then
  echo "Error: C API shared library not found under ${INSTALL_PREFIX}/lib or ${INSTALL_PREFIX}/lib64" >&2
  exit 1
fi

C_API_SMOKE_TEST="${INSTALL_PREFIX}/bin/cuvs_c_dlsym_smoke"
"${CC:-cc}" -std=c11 -Wall -Wextra -Werror \
  "${SCRIPT_DIR}/standalone_c/dlsym_smoke.c" -ldl -o "${C_API_SMOKE_TEST}"
LD_LIBRARY_PATH="$(dirname "${C_API_LIBRARY}"):${LD_LIBRARY_PATH:-}" \
  "${C_API_SMOKE_TEST}" "${C_API_LIBRARY}"

rapids-logger "Run C API tests"
ls -l "${INSTALL_PREFIX}"
cd "$INSTALL_PREFIX"/bin/gtests/libcuvs
ctest -j8 --output-on-failure

rapids-logger "C API tests completed successfully"

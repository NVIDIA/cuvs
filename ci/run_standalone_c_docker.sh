#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# Build and run the standalone C Docker image, then copy the output tarball
# to the current directory so CI artifact upload finds it at libcuvs_c.tar.gz.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CUDA_VERSION="${CUDA_VERSION:-13.0}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
BUILD_OUTPUT_DIR="${BUILD_OUTPUT_DIR:-./build}"
IMAGE_NAME="${IMAGE_NAME:-cuvs-standalone-c}"

# Optional: pass --build-tests to build and install C library tests
BUILD_ARGS=()
while [[ "${1:-}" == --* ]]; do
  BUILD_ARGS+=("$1")
  shift
done

mkdir -p "${BUILD_OUTPUT_DIR}"
BUILD_OUTPUT_DIR_ABS="$(cd "${BUILD_OUTPUT_DIR}" && pwd)"

echo "Building Docker image ${IMAGE_NAME} (CUDA ${CUDA_VERSION}, Python ${PYTHON_VERSION})..."
docker build -f ci/standalone_c/Dockerfile.standalone_c \
  --build-arg CUDA_VERSION="${CUDA_VERSION}" \
  --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
  -t "${IMAGE_NAME}" \
  .

# Pass through sccache-dist and AWS env so the build inside the container can use distributed cache (CI).
DOCKER_ENV=()
for var in AWS_REGION AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN RAPIDS_AUX_SECRET_1 RAPIDS_ARTIFACTS_DIR; do
  if [[ -n "${!var:-}" ]]; then
    DOCKER_ENV+=(-e "${var}=${!var}")
  fi
done

echo "Running standalone C build in container..."
docker run --rm \
  -v "${REPO_ROOT}:/workspace" \
  -v "${BUILD_OUTPUT_DIR_ABS}:/build" \
  "${DOCKER_ENV[@]}" \
  "${IMAGE_NAME}" \
  "${BUILD_ARGS[@]}"

if [[ ! -f "${BUILD_OUTPUT_DIR_ABS}/libcuvs_c.tar.gz" ]]; then
  echo "Expected tarball not found at ${BUILD_OUTPUT_DIR_ABS}/libcuvs_c.tar.gz" >&2
  exit 1
fi

cp -v "${BUILD_OUTPUT_DIR_ABS}/libcuvs_c.tar.gz" "${REPO_ROOT}/libcuvs_c.tar.gz"
echo "Copied libcuvs_c.tar.gz to ${REPO_ROOT}/libcuvs_c.tar.gz"

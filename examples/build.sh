#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# cuvs empty project template build script

# Abort script on first error
set -e

NUMARGS=$#
ARGS=$*

function hasArg {
    (( NUMARGS != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

if hasArg clean; then
  rm -rf c/build
  rm -rf cpp/build
  exit 0
fi

function gpuArch {

    if hasArg --allgpuarch && [[ -n $(echo "$ARGS" | { grep -E "\-\-gpu\-arch" || true; } ) ]]; then
        echo "Error: Cannot specify both --gpu-arch and --allgpuarch"
        echo "Use either:"
        echo "  --gpu-arch=\"80-real;90-real\"    (for specific architectures)"
        echo "  --allgpuarch        (for all supported architectures)"
        exit 1
    fi

    if [[ $(echo "$ARGS" | { grep -Eo "\-\-gpu\-arch" || true; } | wc -l ) -gt 1 ]]; then
        echo "Error: Multiple --gpu-arch options were provided. Please combine architectures into a single option."
        echo "Instead of: --gpu-arch=80-real --gpu-arch=90-real"
        echo "Use:       --gpu-arch=\"80-real;90-real\""
        exit 1
    fi

    if [[ -n $(echo "$ARGS" | { grep -E "\-\-gpu\-arch" || true; } ) ]]; then
        GPU_ARCH_ARG=$(echo "$ARGS" | { grep -Eo "\-\-gpu\-arch=.+( |$)" || true; })
        if [[ -n ${GPU_ARCH_ARG} ]]; then
            # Extract just the architecture value
            echo "${GPU_ARCH_ARG}" | sed -e 's/--gpu-arch=//' -e 's/ .*//'
            return
        fi
    fi

    # Handle --allgpuarch
    if hasArg --allgpuarch; then
        echo "RAPIDS"
        return
    fi

    # Default to NATIVE
    echo "NATIVE"
}

# Set up build configuration
PARALLEL_LEVEL=${PARALLEL_LEVEL:=$(nproc)}
BUILD_TYPE=Release
CUVS_REPO_REL=""
EXTRA_CMAKE_ARGS=()


CUVS_CMAKE_CUDA_ARCHITECTURES=$(gpuArch)
case ${CUVS_CMAKE_CUDA_ARCHITECTURES} in
    "RAPIDS") echo "Building for *ALL* supported GPU architectures..." ;;
    "NATIVE") echo "Building for the architecture of the GPU in the system..." ;;
    *) echo "Building for specified GPU architectures: ${CUVS_CMAKE_CUDA_ARCHITECTURES}" ;;
esac

# Root of examples
EXAMPLES_DIR=$(dirname "$(realpath "$0")")

if [[ ${CUVS_REPO_REL} != "" ]]; then
  CUVS_REPO_PATH=$(readlink -f "${CUVS_REPO_REL}")
  EXTRA_CMAKE_ARGS+=("-DCPM_cuvs_SOURCE=${CUVS_REPO_PATH}")
else
  LIB_BUILD_DIR=${LIB_BUILD_DIR:-$(readlink -f "${EXAMPLES_DIR}/../cpp/build")}
  EXTRA_CMAKE_ARGS+=("-Dcuvs_ROOT=${LIB_BUILD_DIR}")
fi

################################################################################
# Add individual libcuvs examples build scripts down below

build_example() {
  example_dir=${1}
  example_dir="${EXAMPLES_DIR}/${example_dir}"
  build_dir="${example_dir}/build"

  # Configure
  cmake -S "${example_dir}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -DCUVS_NVTX=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${CUVS_CMAKE_CUDA_ARCHITECTURES}" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  "${EXTRA_CMAKE_ARGS[@]}"
  # Build
  cmake --build "${build_dir}" -j"${PARALLEL_LEVEL}"
}

build_example c
build_example cpp

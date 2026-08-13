#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -e -u -o pipefail

echo "Starting Panama FFM API bindings generation ..."
REPODIR=$(cd "$(dirname "$0")"; cd ../../ ; pwd)
CURDIR=$(cd "$(dirname "$0")"; pwd)
TARGET_PACKAGE="com.nvidia.cuvs.internal.panama"

TARGET_DIR="targets/x86_64-linux/include"
if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/${TARGET_DIR}" ]; then
  CUDA_INCLUDE_DIR="${CONDA_PREFIX}/${TARGET_DIR}"
elif [ -d "/usr/local/cuda/${TARGET_DIR}" ]; then
  CUDA_INCLUDE_DIR="/usr/local/cuda/${TARGET_DIR}"
else
  echo "Couldn't find a suitable CUDA include directory."
  exit 1
fi

PATH="$(pwd)/jextract-22/bin/:${PATH}"
export PATH

if [[ $(command -v jextract) == "" ]];
then
  JEXTRACT_FILENAME="openjdk-22-jextract+6-47_linux-x64_bin.tar.gz"
  JEXTRACT_DOWNLOAD_URL="https://download.java.net/java/early_access/jextract/22/6/${JEXTRACT_FILENAME}"
  echo "jextract doesn't exist. Downloading it from $JEXTRACT_DOWNLOAD_URL.";
  wget -c $JEXTRACT_DOWNLOAD_URL
  tar -xvf ./"${JEXTRACT_FILENAME}"
  echo "jextract downloaded to $(pwd)/jextract-22"
fi

# Use Jextract utility to generate panama bindings
jextract \
 --include-dir "${REPODIR}"/java/internal/build/bindings/include/ \
 --include-dir "${CUDA_INCLUDE_DIR}" \
 --output "${REPODIR}/java/cuvs-java/src/main/java22/" \
 --target-package ${TARGET_PACKAGE} \
 --library cuvs_c \
 --library cudart \
 "${CURDIR}"/headers.h

echo "Panama FFM API bindings generation done"

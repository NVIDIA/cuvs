#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e -u -o pipefail

echo "Starting Panama FFM API bindings generation ..."
REPODIR=$(cd "$(dirname "$0")"; cd ../../ ; pwd)
CURDIR=$(cd "$(dirname "$0")"; pwd)
TARGET_PACKAGE="com.nvidia.cuvs.internal.panama"

ARCH=$(uname -m)
case "${ARCH}" in
  x86_64) CUDA_TARGET_ARCH="x86_64-linux" ;;
  aarch64) CUDA_TARGET_ARCH="sbsa-linux" ;;
  *)
    echo "Unsupported architecture for CUDA include directory lookup: ${ARCH}"
    exit 1
    ;;
esac
TARGET_DIR="targets/${CUDA_TARGET_ARCH}/include"
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
  # Checksums are published at ${JEXTRACT_DOWNLOAD_URL}.sha256
  case "${ARCH}" in
    x86_64)
      JEXTRACT_ARCH="linux-x64"
      JEXTRACT_SHA256="a6a42d5b5f4bff119455daadd1ccec9389b2554aea3342f391577843769cc7ec"
      ;;
    aarch64)
      JEXTRACT_ARCH="linux-aarch64"
      JEXTRACT_SHA256="e218579a43dd1cb77e94cdbfa5b7b30eaed682aef8f40300c9916e59b1194fa7"
      ;;
    *)
      echo "Unsupported architecture for jextract download: ${ARCH}"
      exit 1
      ;;
  esac
  JEXTRACT_FILENAME="openjdk-22-jextract+6-47_${JEXTRACT_ARCH}_bin.tar.gz"
  JEXTRACT_DOWNLOAD_URL="https://download.java.net/java/early_access/jextract/22/6/${JEXTRACT_FILENAME}"
  echo "jextract doesn't exist. Downloading it from $JEXTRACT_DOWNLOAD_URL.";
  for attempt in 1 2 3 4 5; do
    rm -f "${JEXTRACT_FILENAME}"
    if wget --tries=1 --timeout=60 -O "${JEXTRACT_FILENAME}" "${JEXTRACT_DOWNLOAD_URL}" \
      && echo "${JEXTRACT_SHA256}  ${JEXTRACT_FILENAME}" | sha256sum -c -; then
      break
    elif [[ ${attempt} == 5 ]]; then
      rm -f "${JEXTRACT_FILENAME}"
      echo "Failed to download jextract after ${attempt} attempts."
      exit 1
    fi
    sleep $((attempt * 15))
  done
  tar -xvf ./"${JEXTRACT_FILENAME}"
  echo "jextract downloaded to $(pwd)/jextract-22"
fi

OUTPUT_DIR="${REPODIR}/java/cuvs-java/src/main/java22"

# jextract overwrites files but never removes them, so bindings for symbols that were
# renamed or dropped from the headers would linger and still compile into the jar. Start
# from an empty directory. Every file is rewritten on each run anyway, so this costs
# nothing in build time.
rm -rf "${OUTPUT_DIR:?}/${TARGET_PACKAGE//.//}"

# Use Jextract utility to generate panama bindings
jextract \
 --include-dir "${REPODIR}"/java/internal/build/bindings/include/ \
 --include-dir "${CUDA_INCLUDE_DIR}" \
 --output "${OUTPUT_DIR}/" \
 --target-package ${TARGET_PACKAGE} \
 --library cuvs_c \
 --library cudart \
 "${CURDIR}"/headers.h

echo "Panama FFM API bindings generation done"

#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# TODO: Remove this argument-handling when build and test workflows are separated,
#       and test_java.sh no longer calls build_java.sh
#       ref: https://github.com/rapidsai/cuvs/issues/868
EXTRA_BUILD_ARGS=()
if [[ "${1:-}" == "--run-java-tests" ]]; then
  EXTRA_BUILD_ARGS+=("--run-java-tests")
fi

# Always build cuvs-java when running the pipeline
EXTRA_BUILD_ARGS+=("--build-cuvs-java")

# shellcheck disable=SC1091
. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Generate Java testing dependencies"

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file-key java \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n java

# Temporarily allow unbound variables for conda activation.
set +u
conda activate java
set -u

rapids-print-env

# Locates the libcuvs.so file path and appends it to LD_LIBRARY_PATH
rapids-logger "Find libcuvs so file and prepend paths to LD_LIBRARY_PATH"

CONDA_PKG_CACHE_DIR="/opt/conda/pkgs" # comes from `conda info`. Dont know if this ever changes.
if [ -d "$CONDA_PKG_CACHE_DIR" ]; then
  echo "==> Directory '$CONDA_PKG_CACHE_DIR' exists."
  LIBCUVS_SO_FILE="libcuvs.so"
  LIBCUVS_PATH=$(find $CONDA_PKG_CACHE_DIR -name $LIBCUVS_SO_FILE)
  if [ -z "$LIBCUVS_PATH" ]; then
    echo "==> Could not find the so file. Not updating LD_LIBRARY_PATH"
    exit 1
  else
    LIBCUVS_DIR=$(dirname "$LIBCUVS_PATH")
    export LD_LIBRARY_PATH="$LIBCUVS_DIR:$LD_LIBRARY_PATH"
    echo "LD_LIBRARY_PATH is: $LD_LIBRARY_PATH"
  fi
else
  echo "==> Directory '$CONDA_PKG_CACHE_DIR' does not exist. Not updating LD_LIBRARY_PATH"
  exit 1
fi

rapids-logger "Run Java build"

bash ./build.sh "${EXTRA_BUILD_ARGS[@]}"

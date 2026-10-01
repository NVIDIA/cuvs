#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcuvs cuvs --cuda "$RAPIDS_CUDA_VERSION")")

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-dependency-file-generator \
  --output conda \
  --file-key rust \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
  --prepend-channel "${CPP_CHANNEL}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n rust

# seeing failures on activating the environment here on unbound locals
# apply workaround from https://github.com/conda/conda/issues/8186#issuecomment-532874667
set +eu
conda activate rust
set -eu

source rapids-configure-sccache

# Don't use the build cluster because conda rust toolchains are too large
export SCCACHE_NO_DIST_COMPILE=1
export SCCACHE_S3_KEY_PREFIX="cuvs-rs/${RAPIDS_CONDA_ARCH}/cuda${RAPIDS_CUDA_VERSION%%.*}"
export SCCACHE_S3_PREPROCESSOR_CACHE_KEY_PREFIX="cuvs-rs/${RAPIDS_CONDA_ARCH}/cuda${RAPIDS_CUDA_VERSION%%.*}/wheel/preprocessor-cache"
export SCCACHE_S3_USE_PREPROCESSOR_CACHE_MODE=true

rapids-print-env

rapids-logger "Begin Rust build"

sccache --stop-server 2>/dev/null || true

bash ./build.sh rust

sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true

# Also test out that we can publish cuvs-sys via a dry-run
pushd ./rust/cuvs-sys
cargo publish --dry-run
popd

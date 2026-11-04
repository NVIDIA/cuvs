#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Usage: ci/test_cpp.sh [SHARD NUM_SHARDS]
# With SHARD and NUM_SHARDS (1 <= SHARD <= NUM_SHARDS), only every NUM_SHARDS-th libcuvs test is
# run, starting from test number SHARD, so the tests can be split across several CI jobs.
SHARD=${1:-1}
NUM_SHARDS=${2:-1}

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcuvs cuvs --cuda "$RAPIDS_CUDA_VERSION")")

rapids-logger "Generate C++ testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" \
  --prepend-channel "${CPP_CHANNEL}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

# CI provides CUDA_CACHE_PATH through the reusable workflow's cache-environment input.
# So that we can re-use the CUDA driver's on-disk JIT cache between runs.
CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-.cache/cuda-jit}"
if [[ "${CUDA_CACHE_PATH}" != /* ]]; then
  CUDA_CACHE_PATH="$(realpath -m "${CUDA_CACHE_PATH}")"
fi
export CUDA_CACHE_PATH
mkdir -p "${CUDA_CACHE_PATH}"
echo "CUDA JIT cache ${CUDA_CACHE_PATH}: $(find "${CUDA_CACHE_PATH}" -type f | wc -l) files before the tests"

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

# RAPIDS_DATASET_ROOT_DIR is used by test scripts
RAPIDS_DATASET_ROOT_DIR=${RAPIDS_TESTS_DIR}/dataset
export RAPIDS_DATASET_ROOT_DIR
./ci/get_test_data.sh --NEIGHBORS_ANN_VAMANA_TEST

uname -a
rapids-mamba-retry install --yes -n test -c conda-forge gdb
cd "$CONDA_PREFIX"/bin/gtests/libcuvs
set +e
df -h /dev/shm
if [[ "${SHARD}" == "2" ]]; then
  rapids-logger "DIAG: IVF_PQ build_host_input_search/* (all types)"
  gdb -q -batch -ex run -ex bt --args ./NEIGHBORS_ANN_IVF_PQ_TEST --gtest_filter='*build_host_input_search*:*build_extend_search/2*'
  rapids-logger "DIAG: full NEIGHBORS_ANN_IVF_PQ_TEST"
  ./NEIGHBORS_ANN_IVF_PQ_TEST --gtest_brief=1
fi
if [[ "${SHARD}" == "4" ]]; then
  F='KmeansSNMGTests/KmeansSNMGTestF.Result/0'
  rapids-logger "DIAG: VMM probe"
  python "${OLDPWD}/ci/vmm_probe.py"
  rapids-logger "DIAG: KMEANS_MG NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL"
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ALLOC ./CLUSTER_KMEANS_MG_TEST --gtest_filter="$F" 2>&1 | grep -v "Channel [0-9]*/64"
  rapids-logger "DIAG: KMEANS_MG NCCL_CUMEM_ENABLE=0 NCCL_SHM_DISABLE=1"
  NCCL_DEBUG=WARN NCCL_CUMEM_ENABLE=0 NCCL_SHM_DISABLE=1 ./CLUSTER_KMEANS_MG_TEST --gtest_filter="$F"
  rapids-logger "DIAG: KMEANS_MG NCCL_CUMEM_ENABLE=0 NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1"
  NCCL_DEBUG=WARN NCCL_CUMEM_ENABLE=0 NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 ./CLUSTER_KMEANS_MG_TEST --gtest_filter="$F"
  rapids-logger "DIAG: KMEANS_MG NCCL_CUMEM_ENABLE=0 NCCL_NCHANNELS_PER_NET_PEER=1 NCCL_MAX_NCHANNELS=1"
  NCCL_DEBUG=WARN NCCL_CUMEM_ENABLE=0 NCCL_MAX_NCHANNELS=1 ./CLUSTER_KMEANS_MG_TEST --gtest_filter="$F"
fi
exit 1

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Run Python build utilities tests (once, in the first shard)
if [[ "${SHARD}" == "1" ]]; then
  rapids-logger "Run libcuvs Python build utilities tests"
  pytest cpp/tests/python
fi

# Run libcuvs gtests from libcuvs-tests package
rapids-logger "Run libcuvs tests (shard ${SHARD} of ${NUM_SHARDS})"
pushd "$CONDA_PREFIX"/bin/gtests/libcuvs
timeout -v --signal=SIGINT --kill-after=60s 100m ctest -j8 --output-on-failure -I "${SHARD},,${NUM_SHARDS}"
popd

# Every kernel that was not in the restored cache and had to be linked adds files.
echo "CUDA JIT cache ${CUDA_CACHE_PATH}: $(find "${CUDA_CACHE_PATH}" -type f | wc -l) files ($(du -sh "${CUDA_CACHE_PATH}" | cut -f1)) after the tests"

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

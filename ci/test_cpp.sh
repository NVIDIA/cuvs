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

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

# RAPIDS_DATASET_ROOT_DIR is used by test scripts
RAPIDS_DATASET_ROOT_DIR=${RAPIDS_TESTS_DIR}/dataset
export RAPIDS_DATASET_ROOT_DIR
./ci/get_test_data.sh --NEIGHBORS_ANN_VAMANA_TEST

rapids-logger "Check CPU resources"
nproc
lscpu | grep -E "^(Architecture|Model name|CPU\(s\)|Thread\(s\) per core|NUMA node\(s\))"
nvidia-smi topo -m || true

# Keep host-side work on the CPUs (and memory) local to the GPU under test. CPU_BIND_CMD
# is prepended to ctest so every test process inherits it. Only done for single-GPU
# runners, and only if the binding is permitted inside the container; otherwise it is empty.
# Prefer numactl (binds CPUs and memory, which matters for pinned host buffers), and fall
# back to taskset (CPUs only) when the GPU reports no NUMA node or numactl isn't usable.
CPU_BIND_CMD=()
if [[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -eq 1 ]]; then
  gpu_bus_id=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader)
  gpu_bus_id=${gpu_bus_id,,}
  gpu_sysfs=/sys/bus/pci/devices/${gpu_bus_id:4} # "00000000:0a:00.0" -> "0000:0a:00.0"
  gpu_numa_node=$(cat "${gpu_sysfs}/numa_node" 2> /dev/null || echo -1)
  gpu_local_cpus=$(cat "${gpu_sysfs}/local_cpulist" 2> /dev/null || true)
  echo "GPU ${gpu_bus_id}: numa_node=${gpu_numa_node} local_cpulist=${gpu_local_cpus:-<unknown>}"
  if ((gpu_numa_node >= 0)) && command -v numactl > /dev/null &&
    numactl --cpunodebind="${gpu_numa_node}" --membind="${gpu_numa_node}" true 2> /dev/null; then
    CPU_BIND_CMD=(numactl --cpunodebind="${gpu_numa_node}" --membind="${gpu_numa_node}")
  elif [[ -n "${gpu_local_cpus}" ]] && command -v taskset > /dev/null &&
    taskset -c "${gpu_local_cpus}" true 2> /dev/null; then
    CPU_BIND_CMD=(taskset -c "${gpu_local_cpus}")
  fi
fi
echo "CPU_BIND_CMD=${CPU_BIND_CMD[*]:-<none>}"
echo "CPUs available after binding: $("${CPU_BIND_CMD[@]}" nproc)"

# OpenMP sizes its thread pool from the visible cores, which in a container is the
# host's core count and not the pod's CPU quota. Cap OMP_NUM_THREADS to the quota (and to
# the CPUs left after binding) so host-side work (HNSW build, NN-Descent, ...) doesn't
# oversubscribe the pod. We deliberately don't set OMP_PROC_BIND: ctest runs several
# processes at once, and 'close' binding would stack every process's thread 0, 1, ... on
# the same first cores.
CPU_QUOTA_CORES=""
if [[ -r /sys/fs/cgroup/cpu.max ]]; then
  # cgroup v2: "<quota|max> <period>"
  read -r cpu_quota cpu_period < /sys/fs/cgroup/cpu.max
  cat /sys/fs/cgroup/cpu.max
  if [[ "${cpu_quota}" != "max" ]]; then
    CPU_QUOTA_CORES=$(((cpu_quota + cpu_period - 1) / cpu_period))
  fi
elif [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]]; then
  # cgroup v1
  cpu_quota=$(</sys/fs/cgroup/cpu/cpu.cfs_quota_us)
  cpu_period=$(</sys/fs/cgroup/cpu/cpu.cfs_period_us)
  echo "cfs_quota_us=${cpu_quota} cfs_period_us=${cpu_period}"
  if ((cpu_quota > 0)); then
    CPU_QUOTA_CORES=$(((cpu_quota + cpu_period - 1) / cpu_period))
  fi
fi
if [[ -z "${OMP_NUM_THREADS:-}" && -n "${CPU_QUOTA_CORES}" ]]; then
  bound_cpus=$("${CPU_BIND_CMD[@]}" nproc)
  export OMP_NUM_THREADS=$((bound_cpus < CPU_QUOTA_CORES ? bound_cpus : CPU_QUOTA_CORES))
fi
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-<unset>}"

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
"${CPU_BIND_CMD[@]}" ctest -j8 --output-on-failure -I "${SHARD},,${NUM_SHARDS}"

popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# DO NOT MERGE: reproducer for the H100 CAGRA Python test hang (#2772, #2732).
# Uses nightly conda packages, repeats the CAGRA tests and the cuvs_bench CLI
# test, and dumps host/GPU stacks when a hang is detected.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Generate Python testing dependencies (nightly packages)"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};dependencies=${RAPIDS_DEPENDENCIES}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

rapids-logger "Create debugging tools environments"
rapids-mamba-retry create --yes -n dbg -c conda-forge gdb py-spy || true
rapids-mamba-retry create --yes -n cudagdb -c conda-forge "cuda-gdb=12.9" "cuda-version=12.9" || true
DBG_BIN=/opt/conda/envs/dbg/bin
CUDA_GDB=$(ls /opt/conda/envs/cudagdb/bin/cuda-gdb 2>/dev/null || true)

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-print-env

rapids-logger "System info"
nvidia-smi
nvidia-smi -q || true
nproc
ulimit -a
env | grep -E "^(CUDA|OMP|NVIDIA|RAPIDS|LD_)" | sort || true
ldconfig -p | grep -E "cudadebugger|libcuda\.so" || true
cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null || true
lscpu || true
cat /sys/fs/cgroup/cpu.max /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null || true
taskset -p $$ || true
free -g || true

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"
DIAG_DIR="${PWD}/hang-diag"
mkdir -p "${DIAG_DIR}"

descendants() {
  local p=$1
  echo "$p"
  for c in $(pgrep -P "$p" 2>/dev/null); do
    descendants "$c"
  done
}

dump_one() {
  local p=$1
  echo "================ PID ${p}: $(tr '\0' ' ' < /proc/"${p}"/cmdline 2>/dev/null | cut -c1-300)"
  grep -E "^(State|Threads|voluntary|nonvoluntary)" /proc/"${p}"/status 2>/dev/null || true
  echo "---- threads (tid comm state utime stime wchan)"
  for t in /proc/"${p}"/task/*; do
    tid=$(basename "$t")
    stat=$(cat "$t"/stat 2>/dev/null) || continue
    # fields after the ')' : state is field 3, utime 14, stime 15
    rest=${stat##*) }
    # shellcheck disable=SC2086
    set -- $rest
    echo "${tid} $(cat "$t"/comm 2>/dev/null) state=$1 utime=${12} stime=${13} wchan=$(cat "$t"/wchan 2>/dev/null)"
  done
  if grep -q python /proc/"${p}"/cmdline 2>/dev/null; then
    echo "---- py-spy dump --native"
    timeout 120 "${DBG_BIN}/py-spy" dump --native --pid "${p}" 2>&1 || true
  fi
  echo "---- gdb thread apply all bt"
  timeout 300 "${DBG_BIN}/gdb" -p "${p}" -batch -nx \
    -ex "set pagination off" -ex "set print thread-events off" \
    -ex "info threads" -ex "thread apply all bt 40" 2>&1 | grep -v "^\[New LWP" || true
}

dump_cuda_gdb() {
  local p=$1
  if [[ -n "${CUDA_GDB}" ]]; then
    echo "---- cuda-gdb ${p}"
    timeout 300 "${CUDA_GDB}" -p "${p}" -batch -nx \
      -ex "set pagination off" \
      -ex "info cuda devices" \
      -ex "info cuda kernels" \
      -ex "info cuda blocks" \
      -ex "info cuda warps" \
      -ex "info cuda threads" \
      -ex "bt" 2>&1 | head -400 || true
  fi
}

dump_diag() {
  local root=$1
  local pids
  pids=$(descendants "${root}")
  echo "########## HANG DIAGNOSTICS $(date -u +%FT%TZ) root=${root} pids: ${pids}"
  ps -eo pid,ppid,stat,wchan:32,etime,time,pcpu,rss,cmd --forest | cut -c1-250 || true
  echo "---- top threads"
  for p in ${pids}; do top -b -n 1 -H -p "${p}" 2>/dev/null | head -40 || true; done
  echo "---- nvidia-smi"
  nvidia-smi || true
  nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,power.draw,memory.used,clocks_throttle_reasons.active --format=csv -lms 500 & local smipid=$!
  sleep 5; kill "${smipid}" 2>/dev/null || true
  nvidia-smi -q -d PERFORMANCE,CLOCK,ECC,PIDS,ROW_REMAPPER 2>&1 | head -200 || true
  for p in ${pids}; do dump_one "${p}"; done
  # cuda-gdb last: attaching may disturb the process
  for p in ${pids}; do
    if grep -qE "python|ANN_BENCH" /proc/"${p}"/cmdline 2>/dev/null; then dump_cuda_gdb "${p}"; fi
  done
  echo "---- dmesg tail"
  dmesg 2>/dev/null | tail -30 || true
  echo "########## END HANG DIAGNOSTICS"
}

HANGS=0
HANG_LIST=""
# run_watched <name> <stall_seconds> <cmd...>
run_watched() {
  local name=$1 stall=$2
  shift 2
  local log="${DIAG_DIR}/${name}.log"
  local start
  start=$(date +%s)
  echo ">>>>> START ${name}: $*"
  "$@" > "${log}" 2>&1 &
  local pid=$!
  tail -n +1 -F "${log}" 2>/dev/null &
  local tailpid=$!
  local last_size=-1 last_change now size hung=0
  last_change=$(date +%s)
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 5
    size=$(stat -c %s "${log}" 2>/dev/null || echo 0)
    now=$(date +%s)
    if [[ "${size}" != "${last_size}" ]]; then
      last_size=${size}
      last_change=${now}
    elif (( now - last_change > stall )); then
      hung=1
      break
    fi
  done
  sleep 2
  kill "${tailpid}" 2>/dev/null || true
  wait "${tailpid}" 2>/dev/null || true
  local rc=0
  if (( hung )); then
    HANGS=$((HANGS + 1))
    HANG_LIST="${HANG_LIST} ${name}"
    echo "!!!!! HANG DETECTED in ${name} after $(( $(date +%s) - start ))s (no output for ${stall}s)"
    dump_diag "${pid}"
    for p in $(descendants "${pid}"); do kill -9 "${p}" 2>/dev/null || true; done
    wait "${pid}" 2>/dev/null
    rc=124
    echo "---- nvidia-smi after kill"
    sleep 5
    nvidia-smi || true
  else
    wait "${pid}"
    rc=$?
  fi
  echo "<<<<< END ${name}: rc=${rc} elapsed=$(( $(date +%s) - start ))s"
  return ${rc}
}

set +e

CUVS_TESTS_DIR="${PWD}/python/cuvs/cuvs"
BENCH_TESTS_DIR="${PWD}/python/cuvs_bench/cuvs_bench"
PYTEST_ARGS=(-v -p no:cacheprovider -o faulthandler_timeout=240 -o junit_family=xunit2)

# Total budget for the repeat loop (seconds).
BUDGET=$(( 200 * 60 ))
LOOP_START=$(date +%s)
i=0
while (( $(date +%s) - LOOP_START < BUDGET )); do
  i=$((i + 1))
  rapids-logger "Iteration ${i} (hangs so far: ${HANGS}:${HANG_LIST})"

  # Same commands as the regular CI job: the full cuvs suite with coverage, then cuvs_bench.
  pushd "${CUVS_TESTS_DIR}" > /dev/null || exit 1
  run_watched "iter${i}-cuvs-full" 420 python -X faulthandler -m pytest "${PYTEST_ARGS[@]}" \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-cuvs.xml" \
    --cov-config=../.coveragerc \
    --cov=cuvs \
    --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cuvs-coverage.xml" \
    --cov-report=term \
    tests
  popd > /dev/null || exit 1

  pushd "${BENCH_TESTS_DIR}" > /dev/null || exit 1
  run_watched "iter${i}-bench" 420 python -X faulthandler -m pytest "${PYTEST_ARGS[@]}" \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-cuvs.xml" \
    --cov-config=../.coveragerc \
    --cov=cuvs \
    --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cuvs-bench-coverage.xml" \
    --cov-report=term \
    tests
  popd > /dev/null || exit 1

  if (( HANGS >= 2 )); then
    break
  fi
done

rapids-logger "HANG SUMMARY: ${HANGS} hang(s) in ${i} iteration(s):${HANG_LIST}"
if (( HANGS > 0 )); then
  exit 1
fi
exit 0

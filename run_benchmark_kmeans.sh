#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Compare baseline / cuTile / flash-kmeans for one shape or the default sweep.
#
# Usage:
#   export BENCH_CONDA=/path/to/miniforge3
#   export BENCH_ENV_BASE=...
#   export BENCH_ENV_CUTILE=...
#   export BENCH_ENV_FLASH=...
#   export MAX_ITER=5 TOL=1e-4 SEED=42
#   export WARMUP_FIT=1 ITERS_FIT=3 WARMUP_PRED=1 ITERS_PRED=3
#   export BENCH_PHASE=both          # fit | predict | both (default: both)
#   export BENCH_GPU_NAME=rtx_pro_6000  # sweep log tag (e.g. h200 on Hopper)
#   export BENCH_LOG=path/to.log        # optional sweep log override
#   ./run_benchmark_kmeans.sh              # default sweep (M=1M, all D and K below)
#   ./run_benchmark_kmeans.sh N D K          # single shape
#   ./run_benchmark_kmeans.sh fit              # sweep, fit only
#   ./run_benchmark_kmeans.sh predict N D K    # single shape, predict only
#
# Default sweep grid:
#   M (n_samples)  = 1_000_000
#   D (n_features) = 16 64 128 384 768 1024 1536   # GEMM inner (K) dimension
#   K (n_clusters) = 10 100 1000 10000 100000      # GEMM N dimension
#

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${BENCH_CONDA:?set BENCH_CONDA to conda/miniforge root}"
: "${BENCH_ENV_BASE:?set BENCH_ENV_BASE}"
: "${BENCH_ENV_CUTILE:?set BENCH_ENV_CUTILE}"
: "${BENCH_ENV_FLASH:?set BENCH_ENV_FLASH}"
: "${MAX_ITER:?set MAX_ITER}"
: "${SEED:?set SEED}"
: "${WARMUP_FIT:?set WARMUP_FIT}"
: "${ITERS_FIT:?set ITERS_FIT}"
: "${WARMUP_PRED:?set WARMUP_PRED}"
: "${ITERS_PRED:?set ITERS_PRED}"
: "${TOL:?set TOL}"

PHASE="${BENCH_PHASE:-both}"
if [[ $# -ge 1 && $1 =~ ^(fit|predict|both)$ ]]; then
  PHASE=$1
  shift
fi

run_shape() {
  local n=$1 d=$2 k=$3
  echo "=== benchmark M=${n} D=${d} K=${k} phase=${PHASE} ==="
  python3 "$SCRIPT_DIR/benchmark_kmeans.py" --compare \
    --n "$n" --d "$d" --k "$k" \
    --phase "$PHASE" \
    --max-iter "$MAX_ITER" --tol "$TOL" --seed "$SEED" \
    --warmup-fit "$WARMUP_FIT" --iters-fit "$ITERS_FIT" \
    --warmup-pred "$WARMUP_PRED" --iters-pred "$ITERS_PRED"
}

if [[ $# -eq 3 ]]; then
  run_shape "$1" "$2" "$3"
  exit $?
fi

if [[ $# -ne 0 ]]; then
  echo "usage: $0 [fit|predict|both] [N D K]" >&2
  echo "  no args           — sweep all shapes, phase=\${BENCH_PHASE:-both}" >&2
  echo "  fit|predict|both  — optional phase override, then sweep" >&2
  echo "  [phase] N D K     — run one shape" >&2
  exit 2
fi

: "${BENCH_GPU_NAME:?set BENCH_GPU_NAME e.g. rtx_pro_6000}"
BENCH_LOG="${BENCH_LOG:-${SCRIPT_DIR}/benchmark_kmeans_sweep_${BENCH_GPU_NAME}_$(date +%Y%m%d_%H%M%S).log}"
echo "Logging to ${BENCH_LOG}"
exec > >(tee "$BENCH_LOG") 2>&1

M=1000000
D_VALUES=(16 64 128 384 768 1024 1536)
K_VALUES=(10 100 1000 10000 100000)

for d in "${D_VALUES[@]}"; do
  for k in "${K_VALUES[@]}"; do
    run_shape "$M" "$d" "$k" || true
  done
done

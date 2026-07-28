#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

installed_benchmark_location="${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/ann"
devcontainer_benchmark_location="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../cpp/build/latest/bench/ann"

if [[ -d "${installed_benchmark_location}" ]]; then
  benchmark_location="${installed_benchmark_location}"
elif [[ -d "${devcontainer_benchmark_location}" ]]; then
  benchmark_location="${devcontainer_benchmark_location}"
else
  echo "Error: Benchmark location not found. Searched:" >&2
  echo "  - ${installed_benchmark_location}" >&2
  echo "  - ${devcontainer_benchmark_location}" >&2
  exit 1
fi

benchmark="${benchmark_location}/CUVS_KNN_BRUTE_FORCE_NVBENCH"
if [[ ! -x "${benchmark}" ]]; then
  echo "Error: NVBench executable not found: ${benchmark}" >&2
  exit 1
fi

"${benchmark}" \
  --profile \
  --devices 0 \
  -q \
  --axis num_queries=1 \
  --axis num_db_vecs=1024 \
  --axis dim=32 \
  --axis k=10 \
  --axis metric=InnerProduct \
  --axis layout=row_major

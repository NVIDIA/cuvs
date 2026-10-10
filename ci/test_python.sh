#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# DO NOT MERGE: reproducer for the flaky Python IVF result tests
# (test_build_precomputed, test_ivf_pq[inner_product-*], test_filtered_ivf_flat).
# Uses nightly conda packages and the test files from this branch.

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

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

python -m pip install pytest-repeat

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

# Use this branch's test files with the nightly cuvs package.
CUVS_DIR=$(python -c "import cuvs, os; print(os.path.dirname(cuvs.__file__))")
cp -v python/cuvs/cuvs/tests/*.py "${CUVS_DIR}/tests/"
TESTS="${CUVS_DIR}/tests"

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "precomputed vs regular IVF-PQ: tie diagnostics"
python ci/flaky_h/precomp_diag.py --metric inner_product --codebook cluster --n-queries 10000 --iters 100000 --seconds 600
for m in inner_product sqeuclidean euclidean; do
  for c in subspace cluster; do
    python ci/flaky_h/precomp_diag.py --metric "${m}" --codebook "${c}" --n-queries 10000 --iters 100000 --seconds 90
  done
done

rapids-logger "recall distributions"
python ci/flaky_h/recall_diag.py --n 400 --fixed-repeats 50

rapids-logger "pytest --count on the flaky tests"
timeout -v --signal=SIGINT --kill-after=60s 60m pytest \
  -p no:cacheprovider -q -rf --count=300 \
  "${TESTS}/test_ivf_pq.py::test_build_precomputed" \
  "${TESTS}/test_ivf_pq.py::test_ivf_pq" \
  "${TESTS}/test_ivf_flat.py::test_filtered_ivf_flat" 2>&1 | tail -n 200

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

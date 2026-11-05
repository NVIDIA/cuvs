#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

# Delete system libnccl.so to ensure the wheel is used
rm -rf /usr/lib64/libnccl*

LIBCUVS_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcuvs cuvs --cuda "$RAPIDS_CUDA_VERSION")")
CUVS_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python cuvs cuvs --stable --cuda "$RAPIDS_CUDA_VERSION")")

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints test_python "${PIP_CONSTRAINT}"

python -m venv libcuvs-env
. libcuvs-env/bin/activate

rapids-pip-retry install \
    -v \
    --prefer-binary \
    --constraint "${PIP_CONSTRAINT}" \
    "${LIBCUVS_WHEELHOUSE}"/libcuvs*.whl
python -c "import libcuvs; assert (libraries := libcuvs.load_library()) and all(libraries)"
deactivate

# notes:
#
#   * echo to expand wildcard before adding `[test]` requires for pip
#   * just providing --constraint="${PIP_CONSTRAINT}" to be explicit, and because
#     that environment variable is ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install \
    --prefer-binary \
    --constraint "${PIP_CONSTRAINT}" \
    "${LIBCUVS_WHEELHOUSE}"/libcuvs*.whl \
    "$(echo "${CUVS_WHEELHOUSE}"/cuvs*.whl)[test]"

set +e
nproc; free -g; uname -r
T=./python/cuvs/cuvs/tests
timeout -v --signal=SIGINT --kill-after=30s 15m python -m pytest -v -rfE --tb=long "${T}/test_doctests.py"
for f in test_mg_cagra test_mg_ivf_flat test_mg_kmeans; do
  echo "DIAG: ${f}"
  NCCL_DEBUG=INFO timeout -v --signal=SIGINT --kill-after=30s 5m python -m pytest -v -x -rfE --tb=long -p no:cacheprovider "${T}/${f}.py" 2>&1 | grep -vE 'SKIPPED' | head -n 400
done
echo "DIAG: test_mg_kmeans NCCL_CUMEM_ENABLE=0"
NCCL_DEBUG=WARN NCCL_CUMEM_ENABLE=0 timeout -v --signal=SIGINT --kill-after=30s 5m python -m pytest -v -x -rfE --tb=short "${T}/test_mg_kmeans.py" 2>&1 | tail -n 80
echo "DIAG: full suite minus mg, durations"
timeout -v --signal=SIGINT --kill-after=30s 40m python -m pytest -q --durations=40 "${T}" --ignore "${T}/test_mg_kmeans.py" --ignore "${T}/test_mg_cagra.py" --ignore "${T}/test_mg_ivf_flat.py" --ignore "${T}/test_mg_ivf_pq.py" 2>&1 | tail -n 80
exit 1

#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

# Delete system libnccl.so to ensure the wheel is used
rm -rf /usr/lib64/libnccl*

LIBCUVS_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcuvs cuvs --cuda "$RAPIDS_CUDA_VERSION")")
CUVS_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python cuvs cuvs --stable --cuda "$RAPIDS_CUDA_VERSION")")

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints test_python "${PIP_CONSTRAINT}"

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

python -m pytest ./python/cuvs/cuvs/tests

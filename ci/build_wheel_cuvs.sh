#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

package_dir="python/cuvs"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Downloads libcuvs wheels from this current build,
# then ensures 'cuvs' wheel builds always use the 'libcuvs' just built in the same CI run.
#
# env variable 'PIP_CONSTRAINT' is set up by rapids-init-pip. It constrains all subsequent
# 'pip install', 'pip download', etc. calls (except those used in 'pip wheel', handled separately in build scripts)
LIBCUVS_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcuvs cuvs --cuda "$RAPIDS_CUDA_VERSION")")
echo "libcuvs-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBCUVS_WHEELHOUSE}"/libcuvs_*.whl)" >> "${PIP_CONSTRAINT}"

# TODO: move this variable into `ci-wheel`
# Format Python limited API version string
RAPIDS_PY_API="cp${RAPIDS_PY_VERSION//./}"
export RAPIDS_PY_API

ci/build_wheel.sh cuvs ${package_dir} --stable
ci/validate_wheel.sh ${package_dir} "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

RAPIDS_PACKAGE_NAME="$(rapids-artifact-name wheel_python cuvs cuvs --stable --cuda)"
export RAPIDS_PACKAGE_NAME

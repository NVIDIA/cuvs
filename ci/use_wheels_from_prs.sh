#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TEMPORARY: build and test against wheels from an unmerged RAFT PR.
# Do not merge. https://github.com/NVIDIA/raft/pull/3208
#
# Must be sourced after 'rapids-init-pip', which sets up PIP_CONSTRAINT.
# Constraints appended here apply to every later 'pip install' and, via
# ci/build_wheel.sh's --dependency-constraints-txt, to isolated wheel builds.

RAPIDS_PY_CUDA_SUFFIX=$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")

LIBRAFT_WHEELHOUSE=$(
  rapids-get-pr-artifact NVIDIA/raft 3208 cpp wheel
)
PYLIBRAFT_WHEELHOUSE=$(
  rapids-get-pr-artifact NVIDIA/raft 3208 python wheel --pkg_name pylibraft
)

cat >> "${PIP_CONSTRAINT}" <<EOF
libraft-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBRAFT_WHEELHOUSE}"/libraft_*.whl)
pylibraft-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${PYLIBRAFT_WHEELHOUSE}"/pylibraft_*.whl)
EOF

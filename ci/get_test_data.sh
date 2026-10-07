#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

NUMARGS=$#
ARGS=$*
function has_arg {
    local arg="$1"
    (( NUMARGS != 0 )) && (echo " ${ARGS} " | grep -q " ${arg} ")
}

if has_arg -h || has_arg --help; then
    echo "Usage:"
    echo "    $0 [--NEIGHBORS_ANN_VAMANA_TEST]"
    exit 0
fi

DEST=${RAPIDS_DATASET_ROOT_DIR:-"$PWD"}

# get test data for NEIGHBORS_ANN_VAMANA_TEST
if has_arg "--NEIGHBORS_ANN_VAMANA_TEST"; then
    echo "Downloading test data for NEIGHBORS_ANN_VAMANA_TEST"
    echo "Destination: ${DEST}"
    URL_PREFIX=https://data.rapids.ai/cuvs/tests/data
    SUBDIR=neighbors/ann_vamana/randomized_codebooks
    FILE_LIST=(
        384_int8_pq_pivots.bin
        384_int8_pq_pivots.bin_rotation_matrix.bin
        64_float_pq_pivots.bin
        64_float_pq_pivots.bin_rotation_matrix.bin
    )
    for f in "${FILE_LIST[@]}"; do
        wget --no-verbose --directory-prefix="${DEST}/${SUBDIR}" "${URL_PREFIX}/${SUBDIR}/$f"
    done
fi

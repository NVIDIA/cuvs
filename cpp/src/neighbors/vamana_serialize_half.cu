/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vamana_serialize.cuh"

#include <cuda_fp16.h>

namespace cuvs::neighbors::vamana {

CUVS_INST_VAMANA_SERIALIZE(half);

}  // namespace cuvs::neighbors::vamana

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/ivf_sq.hpp>

#include "ivf_sq_serialize.cuh"

namespace cuvs::neighbors::ivf_sq {

CUVS_INST_IVF_SQ_SERIALIZE(uint8_t);

#undef CUVS_INST_IVF_SQ_SERIALIZE

}  // namespace cuvs::neighbors::ivf_sq

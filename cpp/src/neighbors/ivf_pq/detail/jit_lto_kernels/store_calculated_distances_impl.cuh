/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuvs::neighbors::ivf_pq::detail {

template <typename OutT, bool kManageLocalTopK>
__device__ void store_calculated_distances_impl(OutT* _out_scores,
                                                uint32_t* _out_indices,
                                                uint32_t probe_ix,
                                                uint32_t n_probes,
                                                uint32_t query_ix,
                                                uint32_t topk,
                                                uint32_t max_samples,
                                                OutT*& out_scores,
                                                uint32_t*& out_indices)
{
  if constexpr (kManageLocalTopK) {
    // Store topk calculated distances to out_scores (and its indices to out_indices)
    const uint64_t out_offset = probe_ix + n_probes * query_ix;
    out_scores                = _out_scores + out_offset * topk;
    out_indices               = _out_indices + out_offset * topk;
  } else {
    // Store all calculated distances to out_scores
    out_scores = _out_scores + static_cast<uint64_t>(max_samples) * query_ix;
  }
}

}  // namespace cuvs::neighbors::ivf_pq::detail

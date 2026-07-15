/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <type_traits>

#include <cuvs/detail/jit_lto/ivf_pq/compute_similarity_fragments.hpp>
#include <raft/util/cudart_utils.hpp>

namespace cuvs::neighbors::ivf_pq::detail {

template <typename OutT, typename MetricTag>
__device__ OutT get_early_stop_limit_impl(OutT query_kth)
{
  if constexpr (std::is_same_v<MetricTag, tag_metric_euclidean>) {
    return query_kth;
  } else if constexpr (std::is_same_v<MetricTag, tag_metric_inner_product>) {
    return raft::upper_bound<OutT>();
  } else {
    static_assert(sizeof(MetricTag*) == 0, "Invalid MetricTag");
  }
}

}  // namespace cuvs::neighbors::ivf_pq::detail

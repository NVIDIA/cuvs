/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_mdspan.hpp>

#include <cuvs/core/export.hpp>
#include <vector>

namespace CUVS_EXPORT cuvs {
namespace neighbors {
namespace composite {

/**
 * @brief Composite index that searches multiple CAGRA sub-indices and merges results.
 *
 * When the composite index contains multiple sub-indices, the user can set a
 * stream pool in the input raft::resource to enable parallel search across
 * sub-indices for improved performance.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *
 *   auto index0 = cagra::build(res, params, dataset0);
 *   auto index1 = cagra::build(res, params, dataset1);
 *
 *   composite::composite_index<float, uint32_t> composite({&index0, &index1});
 *
 *   // optional: create a stream pool to enable parallel search across sub-indices
 *   size_t n_streams = 2;
 *   raft::resource::set_cuda_stream_pool(handle,
 *                                        std::make_shared<rmm::cuda_stream_pool>(n_streams));
 *
 *   composite.search(handle, search_params, queries, neighbors, distances);
 * @endcode
 */
template <typename T, typename IdxT, typename OutputIdxT = IdxT>
class CUVS_EXPORT composite_index {
 public:
  using value_type        = T;
  using index_type        = IdxT;
  using out_index_type    = OutputIdxT;
  using matrix_index_type = int64_t;

  explicit composite_index(
    std::vector<cuvs::neighbors::cagra::device_padded_index<T, IdxT>*> children)
    : children_(std::move(children))
  {
  }

  /**
   * @brief Search the composite index for the k nearest neighbors.
   *
   * Searches each sub-index independently (optionally in parallel via stream pool),
   * then selects the top-k results across all sub-indices.
   *
   * @param[in] handle raft resource handle
   * @param[in] params CAGRA search parameters
   * @param[in] queries device matrix view of query vectors [n_queries, dim]
   * @param[out] neighbors device matrix view for neighbor indices [n_queries, k]
   * @param[out] distances device matrix view for distances [n_queries, k]
   * @param[in] filter optional filter for search results
   */
  void search(
    const raft::resources& handle,
    const cuvs::neighbors::cagra::search_params& params,
    raft::device_matrix_view<const value_type, matrix_index_type, raft::row_major> queries,
    raft::device_matrix_view<out_index_type, matrix_index_type, raft::row_major> neighbors,
    raft::device_matrix_view<float, matrix_index_type, raft::row_major> distances,
    const cuvs::neighbors::filtering::base_filter& filter =
      cuvs::neighbors::filtering::none_sample_filter{}) const;

  index_type size() const noexcept
  {
    index_type total = 0;
    for (const auto& c : children_) {
      total += c->size();
    }
    return total;
  }

  cuvs::distance::DistanceType metric() const noexcept
  {
    return children_.empty() ? cuvs::distance::DistanceType::L2Expanded
                             : children_.front()->metric();
  }

 private:
  std::vector<cuvs::neighbors::cagra::device_padded_index<T, IdxT>*> children_;
};

}  // namespace composite
}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs

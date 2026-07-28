/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/roaring.hpp>
#include <cuvs/neighbors/common.hpp>

#include <cstdint>
#include <vector>

namespace cuvs::neighbors::filtering {

/**
 * @defgroup roaring_filters Roaring bitmap sample filters
 * @{
 */

/**
 * @brief Filter an index with a compressed Roaring bitmap shared by all
 * queries.
 *
 * The semantic counterpart of @ref bitset_filter with a compressed filter
 * representation: memory scales with filter cardinality and structure
 * instead of dataset size, and search dispatch can exploit the
 * host-known cardinality (no count kernels) and the container structure
 * (direct CSR emission, row gathering) instead of scanning a flat
 * bitset.
 *
 * The filter is non-owning: the referenced `cuvs::core::gpu_roaring`
 * must outlive any search call using this filter.
 *
 * @code{.cpp}
 *   auto roaring = cuvs::core::from_sorted_ids(res, ids, n_ids, n_rows);
 *   auto filter  = cuvs::neighbors::filtering::roaring_filter(roaring);
 *   cuvs::neighbors::brute_force::search(
 *     res, params, index, queries, neighbors, distances, filter);
 * @endcode
 */
struct roaring_filter : public base_filter {
  /** The bitmap to filter on (non-owning). */
  const cuvs::core::gpu_roaring* bitmap_;
  /** Device view used for per-sample membership tests. */
  cuvs::core::roaring_view view_;

  explicit roaring_filter(const cuvs::core::gpu_roaring& bitmap)
    : bitmap_(&bitmap), view_(bitmap.view())
  {
  }

  /** \cond */
#if defined(__CUDACC__)
  __device__ __forceinline__ bool operator()(
    // query index (ignored: one filter shared by all queries)
    const uint32_t query_ix,
    // the index of the current sample
    const uint32_t sample_ix) const
  {
    (void)query_ix;
    return view_.test(sample_ix);
  }
#endif
  /** \endcond */

  FilterType get_filter_type() const override { return FilterType::Roaring; }
};

/**
 * @brief Filter an index with one compressed Roaring bitmap per query.
 *
 * The semantic counterpart of @ref bitmap_filter (logical shape
 * `[n_queries, index->size()]`) with per-query compressed filters: the
 * dense `n_queries x n_rows` bit matrix is never materialized. Search
 * emits the sparse distance structure directly from the per-query
 * containers (per-query cardinalities are known on the host at filter
 * construction, so the CSR indptr is free and no count kernels or
 * device synchronization are needed).
 *
 * The filter is non-owning: the array of `gpu_roaring const*` and every
 * pointee must outlive any search call using this filter.
 *
 * @code{.cpp}
 *   std::vector<const cuvs::core::gpu_roaring*> per_query = {...};
 *   auto filter = cuvs::neighbors::filtering::roaring_matrix_filter(
 *     per_query.data(), n_queries);
 *   cuvs::neighbors::brute_force::search(
 *     res, params, index, queries, neighbors, distances, filter);
 * @endcode
 */
struct roaring_matrix_filter : public base_filter {
  /** Per-query bitmaps, `bitmaps_[i]` filters query `i` (non-owning). */
  const cuvs::core::gpu_roaring* const* bitmaps_;
  uint32_t n_queries_;

  roaring_matrix_filter(const cuvs::core::gpu_roaring* const* bitmaps, uint32_t n_queries)
    : bitmaps_(bitmaps), n_queries_(n_queries)
  {
  }

  /** Sum of per-query cardinalities (the sparse search nnz). */
  [[nodiscard]] uint64_t nnz() const
  {
    uint64_t total = 0;
    for (uint32_t i = 0; i < n_queries_; ++i)
      total += bitmaps_[i]->total_cardinality;
    return total;
  }

  FilterType get_filter_type() const override { return FilterType::RoaringMatrix; }
};

/** @} */

}  // namespace cuvs::neighbors::filtering

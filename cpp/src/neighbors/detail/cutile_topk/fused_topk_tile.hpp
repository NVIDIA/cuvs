/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>
#include <cuvs/detail/jit_lto/cutile_compat.hpp>
#include <cuvs/distance/distance.hpp>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#ifndef CUVS_CUTILE_ENABLED
#define CUVS_CUTILE_ENABLED 0
#endif

namespace cuvs::neighbors::detail {

inline bool is_fused_topk_metric(cuvs::distance::DistanceType metric)
{
  return metric == cuvs::distance::DistanceType::InnerProduct ||
         metric == cuvs::distance::DistanceType::L2Expanded ||
         metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
         metric == cuvs::distance::DistanceType::CosineExpanded;
}

/**
 * Shape-independent eligibility for the initial SM120 small-TopK backend.
 *
 * This is a capability check, not a table of benchmarked workloads. Query and
 * corpus sizes may differ; arbitrary positive feature widths (including tails)
 * are supported. Public callers must separately permit approximate membership
 * and satisfy the runtime, layout, partition, and buffer checks.
 */
inline bool is_fused_topk_shape_supported(int64_t query_rows,
                                          int64_t corpus_rows,
                                          int64_t dimensions,
                                          cuvs::distance::DistanceType metric)
{
  constexpr int64_t limit = std::numeric_limits<int>::max();
  return query_rows > 0 && query_rows <= limit && corpus_rows > 0 && corpus_rows <= limit &&
         dimensions > 0 && dimensions <= limit && is_fused_topk_metric(metric);
}

template <typename DataT = float>
inline bool is_fused_topk_pk2_candidate(int64_t query_rows,
                                        int64_t corpus_rows,
                                        int64_t dimensions,
                                        int64_t logical_k,
                                        cuvs::distance::DistanceType metric,
                                        bool /* all_neighbors */)
{
  static_assert(std::is_same_v<DataT, float> || std::is_same_v<DataT, half>);
  return logical_k == 2 &&
         is_fused_topk_shape_supported(query_rows, corpus_rows, dimensions, metric);
}

template <typename DataT = float>
inline bool is_fused_topk_pk4_candidate(int64_t query_rows,
                                        int64_t corpus_rows,
                                        int64_t dimensions,
                                        int64_t logical_k,
                                        cuvs::distance::DistanceType metric,
                                        bool /* all_neighbors */)
{
  static_assert(std::is_same_v<DataT, float> || std::is_same_v<DataT, half>);
  return (logical_k == 3 || logical_k == 4) &&
         is_fused_topk_shape_supported(query_rows, corpus_rows, dimensions, metric);
}

constexpr int fused_topk_pk2_ceil_div(int extent, int divisor)
{
  return extent / divisor + (extent % divisor != 0);
}

// Validate the original spans before packing can replace the input pointers.
// Query/corpus aliasing is allowed, but neither may alias either output.
template <typename IdxT, typename DataT>
inline bool fused_topk_public_buffers_supported(IdxT* neighbors,
                                                float* distances,
                                                const DataT* queries,
                                                const DataT* corpus,
                                                int64_t query_rows,
                                                int64_t corpus_rows,
                                                int64_t dimensions,
                                                int64_t logical_k)
{
  struct span {
    uintptr_t begin;
    size_t bytes;
  };
  auto make_span =
    [](const void* pointer, int64_t rows, int64_t columns, size_t element_bytes, span& result) {
      constexpr auto limit = static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max());
      const auto begin     = reinterpret_cast<uintptr_t>(pointer);
      if (!pointer || begin % 16 != 0 || rows <= 0 || columns <= 0 ||
          static_cast<uint64_t>(columns) > limit / element_bytes) {
        return false;
      }
      const size_t row_bytes = static_cast<size_t>(columns) * element_bytes;
      if (static_cast<uint64_t>(rows) > limit / row_bytes) { return false; }
      const size_t bytes = static_cast<size_t>(rows) * row_bytes;
      if (begin > std::numeric_limits<uintptr_t>::max() - bytes) { return false; }
      result = {begin, bytes};
      return true;
    };
  span ids{}, values{}, q{}, x{};
  if (!make_span(neighbors, query_rows, logical_k, sizeof(IdxT), ids) ||
      !make_span(distances, query_rows, logical_k, sizeof(float), values) ||
      !make_span(queries, query_rows, dimensions, sizeof(DataT), q) ||
      !make_span(corpus, corpus_rows, dimensions, sizeof(DataT), x)) {
    return false;
  }
  auto overlaps = [](span a, span b) {
    return a.begin <= b.begin ? b.begin - a.begin < a.bytes : a.begin - b.begin < b.bytes;
  };
  return !overlaps(ids, values) && !overlaps(ids, q) && !overlaps(ids, x) && !overlaps(values, q) &&
         !overlaps(values, x);
}

template <typename DataT>
constexpr int fused_topk_padded_dimensions(int dimensions)
{
  constexpr int alignment = 16 / sizeof(DataT);
  if (dimensions <= 0 || dimensions > std::numeric_limits<int>::max() - (alignment - 1)) {
    return dimensions;
  }
  return fused_topk_pk2_ceil_div(dimensions, alignment) * alignment;
}

// Increase parallelism when query tiles alone cover less than half the GPU.
// This chooses launch geometry, not backend eligibility. It applies equally to
// graph/search, both input types, and every supported metric and feature width.
constexpr int fused_topk_public_num_splits(int query_rows, int corpus_rows, int sm_count)
{
  if (query_rows <= 0 || corpus_rows <= 0 || sm_count <= 0) { return 0; }
  const int query_tiles   = fused_topk_pk2_ceil_div(query_rows, 64);
  const int minimum_tiles = sm_count > 1 ? sm_count / 2 : 1;
  if (query_tiles >= minimum_tiles) { return 1; }
  const int target_tiles = sm_count < 64 ? sm_count * 2 : 128;
  const int desired      = fused_topk_pk2_ceil_div(target_tiles, query_tiles);
  const int corpus_tiles = fused_topk_pk2_ceil_div(corpus_rows, 128);
  return desired < corpus_tiles ? desired : corpus_tiles;
}

constexpr size_t fused_topk_align_workspace(size_t bytes) { return (bytes + 15) & ~size_t{15}; }

// Separate refinement repays its extra launch only above the broad low-D
// region for half PK2. Dimensions describe the actual (possibly padded) input.
// Architecture and ABI eligibility are checked separately by the launcher.
template <int PhysicalK, typename DataT>
constexpr bool fused_topk_separate_refinement_eligible(int dimensions, int num_splits)
{
  static_assert(PhysicalK == 2 || PhysicalK == 4);
  if (dimensions <= 0 || num_splits <= 0) { return false; }
  if constexpr (PhysicalK == 4 && std::is_same_v<DataT, half>) { return true; }
  if (num_splits == 1) { return false; }
  if constexpr (PhysicalK == 2 && std::is_same_v<DataT, half>) { return dimensions > 64; }
  return true;
}

// Overflow-safe ceil division for the 256-thread merge, with a bounded grid.
// The kernel uses a 64-bit grid-stride loop independently of public index width.
constexpr int fused_topk_merge_block_count(int query_rows)
{
  if (query_rows <= 0) { return 0; }
  const int blocks = query_rows / 256 + (query_rows % 256 != 0);
  return blocks < 65535 ? blocks : 65535;
}

template <int PhysicalK, typename IdxT>
constexpr size_t fused_topk_workspace_bytes(IdxT query_rows, int num_splits, int logical_k)
{
  if (query_rows <= 0 || num_splits <= 0 || logical_k <= 0 || logical_k > PhysicalK) { return 0; }
  const auto elements =
    static_cast<size_t>(query_rows) * static_cast<size_t>(num_splits) * PhysicalK;
  const auto index_bytes = elements * sizeof(int);
  if (num_splits > 1 || logical_k != PhysicalK) {
    return fused_topk_align_workspace(index_bytes) + elements * sizeof(float);
  }
  if constexpr (std::is_same_v<IdxT, int64_t>) { return index_bytes; }
  return 0;
}

template <typename IdxT>
constexpr size_t fused_topk_pk2_workspace_bytes(IdxT query_rows, int num_splits)
{
  return fused_topk_workspace_bytes<2>(query_rows, num_splits, 2);
}

template <typename IdxT>
constexpr size_t fused_topk_pk4_workspace_bytes(IdxT query_rows, int num_splits, int logical_k)
{
  return fused_topk_workspace_bytes<4>(query_rows, num_splits, logical_k);
}

#if CUVS_CUTILE_ENABLED
/** Compilation support is broader; automatic brute-force promotion is initially SM120-only. */
CUVS_EXPORT bool is_fused_topk_pk2_auto_dispatch_available();
CUVS_EXPORT bool is_fused_topk_pk4_auto_dispatch_available();

template <typename IdxT, typename DataT = float>
bool try_fused_topk_pk2_tile(IdxT* neighbors,
                             float* distances,
                             const DataT* queries,
                             const DataT* corpus,
                             IdxT query_rows,
                             IdxT corpus_rows,
                             IdxT dimensions,
                             int num_splits,
                             cuvs::distance::DistanceType metric,
                             cudaStream_t stream,
                             void* workspace);

template <typename IdxT, typename DataT = float>
bool try_fused_topk_pk4_tile(IdxT* neighbors,
                             float* distances,
                             const DataT* queries,
                             const DataT* corpus,
                             IdxT query_rows,
                             IdxT corpus_rows,
                             IdxT dimensions,
                             IdxT logical_k,
                             int num_splits,
                             cuvs::distance::DistanceType metric,
                             cudaStream_t stream,
                             void* workspace);
#else
inline bool is_fused_topk_pk2_auto_dispatch_available() { return false; }
inline bool is_fused_topk_pk4_auto_dispatch_available() { return false; }

template <typename IdxT, typename DataT = float>
bool try_fused_topk_pk2_tile(IdxT*,
                             float*,
                             const DataT*,
                             const DataT*,
                             IdxT,
                             IdxT,
                             IdxT,
                             int,
                             cuvs::distance::DistanceType,
                             cudaStream_t,
                             void*)
{
  return false;
}

template <typename IdxT, typename DataT = float>
bool try_fused_topk_pk4_tile(IdxT*,
                             float*,
                             const DataT*,
                             const DataT*,
                             IdxT,
                             IdxT,
                             IdxT,
                             IdxT,
                             int,
                             cuvs::distance::DistanceType,
                             cudaStream_t,
                             void*)
{
  return false;
}
#endif

}  // namespace cuvs::neighbors::detail

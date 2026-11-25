/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>
#include <cuvs/distance/distance.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cstdint>
#include <type_traits>
#include <vector>

namespace CUVS_EXPORT cuvs {
namespace cluster::kmeans::detail::ivf_assign {

/**
 * IVF nearest-centroid assignment with list-grouped GEMMs (internal).
 *
 * Approximate top-1 assignment of a large batch of vectors to a fixed set of `n_centroids`
 * centroids: an IVF-Flat style search over the centroids, organized by list. The centroids are
 * grouped into `n_lists` lists (about sqrt(n_centroids)) by balanced k-means. A batch is assigned
 * by (1) scoring the list centers and taking each vector's `n_probes` best lists, (2) bucketing the
 * (list, vector) pairs by list, (3) for each list gathering its vectors and running one dense TF32
 * GEMM against that list's centroids plus a row argmax, and (4) merging each vector's per-list
 * winners. The cost per vector is about (n_lists + n_probes * n_centroids / n_lists) * dim
 * multiply-adds on tensor cores, versus n_centroids * dim for an exhaustive scan.
 *
 * The result is approximate: a vector whose nearest centroid is in a list it does not probe gets
 * the best centroid among the lists it does probe. There is no worst-case bound on how often
 * that happens; it depends on the data, and `n_lists` and `n_probes` trade accuracy for speed.
 * Probing every list (`n_probes >= n_lists`) is an exhaustive search and gives the exact result
 * (in TF32 arithmetic).
 *
 * The index is built once from a fixed set of centroids and reused for any number of `assign`
 * calls (for example, one per batch).
 */

struct index_params {
  /**
   * Number of lists the centroids are grouped into. 0 means round(sqrt(n_centroids)). More,
   * smaller lists make each probe cheaper but the coarse scoring step and the choice of which
   * lists to probe more expensive.
   */
  uint32_t n_lists = 0;
  /** Iterations of cuVS balanced k-means used to group the centroids. */
  uint32_t kmeans_n_iters = 10;
  /**
   * Each list's GEMM is rounded up to a multiple of this many centroids (the extra columns are
   * computed and ignored). 256 matches the tensor-core tile size and was fastest on H100.
   */
  uint32_t tile_align = 256;
  /**
   * L2Expanded / L2SqrtExpanded: nearest by L2. CosineExpanded: nearest by angle (the centroids
   * are normalized; vectors need not be). InnerProduct: largest inner product.
   */
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded;
};

/** Which implementation of the per-list step `assign` uses. */
enum class assign_algo {
  /** The fused kernel when it can run (see `assign`), otherwise the GEMM loop. */
  AUTO = 0,
  /** The fused kernel; `assign` throws if it cannot run on this device or input. */
  FUSED = 1,
  /** One dense GEMM per probed list, which synchronizes the stream once. */
  GEMM_LOOP = 2
};

struct search_params {
  /**
   * Number of lists probed per vector. More probes make the assignment more accurate and slower;
   * the cost of the per-list step grows linearly with it, as does the scratch memory
   * (batch * n_probes entries). Values above the index's `n_lists` are clamped to it, which
   * makes the search exhaustive.
   *
   * 0 selects 16 (or `n_lists`, if smaller).
   * The default is a rule of thumb, not a guarantee; tune it against the accuracy you need.
   */
  uint32_t n_probes = 0;
  /**
   * Implementation of the per-list step. Both give the same labels up to TF32 rounding order, so
   * the default needs no tuning; the other values are for testing and for comparing speed.
   */
  assign_algo algo = assign_algo::AUTO;
};

static_assert(std::is_aggregate_v<index_params>);
static_assert(std::is_aggregate_v<search_params>);

struct index {
  index(raft::resources const& res);
  index(index&&)            = default;
  index& operator=(index&&) = default;

  uint32_t n_centroids = 0;
  uint32_t dim         = 0;
  uint32_t n_lists     = 0;
  uint32_t tile_align  = 0;
  /** True when scores are x.c - |c|^2/2 (L2); false when they are plain x.c. */
  bool l2 = false;

  /** [n_lists, dim] list centers. */
  raft::device_matrix<float, int64_t, raft::row_major> list_centers;
  /** [n_lists] |center|^2/2 (L2 only). */
  raft::device_vector<float, int64_t> list_bias;
  /**
   * [n_centroids + tile_align, dim] centroids, each list contiguous, trailing rows zero. The values
   * are rounded to TF32 (what the tensor cores use), so they differ from the input centroids in the
   * low 13 mantissa bits.
   */
  raft::device_matrix<float, int64_t, raft::row_major> perm_centroids;
  /** [n_centroids + tile_align] |centroid|^2/2 in the same order (L2 only). */
  raft::device_vector<float, int64_t> perm_bias;
  /** [n_centroids] original centroid id of each row of perm_centroids. */
  raft::device_vector<int32_t, int64_t> perm_id;
  /** Host: first row of each list in perm_centroids, and its size. */
  std::vector<int32_t> list_offset;
  std::vector<int32_t> list_size;
  /**
   * Device copies for the fused assignment kernel: first row of each list [n_lists + 1], size of
   * each list [n_lists], and the list ids by decreasing size [n_lists].
   */
  raft::device_vector<int32_t, int64_t> list_offset_d;
  raft::device_vector<int32_t, int64_t> list_size_d;
  raft::device_vector<int32_t, int64_t> list_order_d;
};

/** Groups `centroids` ([n_centroids, dim], f32) into lists. */
auto build(raft::resources const& res,
           const index_params& params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> centroids) -> index;

/**
 * Writes the id (row of the `centroids` passed to `build`) of each query's approximately nearest
 * centroid into `labels`. Enqueued on the stream of `res`.
 *
 * The queries must be 16-byte aligned (any allocation of rows whose
 * dimension is a multiple of 4 is). On GPUs of compute capability 8.0 or newer the assignment runs
 * in a single fused kernel that needs no host synchronization and no scratch proportional to the
 * largest list. Otherwise it uses one dense GEMM per probed list, which synchronizes the stream
 * once. Both compute the same TF32 scores and give the same labels up to rounding order. For large
 * lists (about 1000 or more centroids) and large batches the per-list GEMMs can be faster than the
 * fused kernel on Hopper, because the fused kernel does not use wgmma. The fused kernel steps
 * through `dim` 32 at a time and zero-fills the last step, so a dimension that is not a multiple of
 * 32 wastes some tensor-core work (about a quarter for `dim` = 100).
 *
 * The index is only read, so the same index can be used with different `params` without being
 * rebuilt.
 */
void assign(raft::resources const& res,
            const search_params& params,
            const index& idx,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_vector_view<uint32_t, int64_t> labels);

}  // namespace cluster::kmeans::detail::ivf_assign
}  // namespace CUVS_EXPORT cuvs

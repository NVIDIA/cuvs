/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/cuda_fp16.hpp>
#include <cuvs/core/export.hpp>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cstdint>

namespace CUVS_EXPORT cuvs {
namespace neighbors {
namespace experimental {
namespace gpipnn {

/**
 * @defgroup gpipnn_cpp_index_params GPiPNN graph build parameters
 * @{
 */

/**
 * @brief Parameters for the GPiPNN proximity graph builder.
 *
 * GPiPNN builds a navigable proximity graph without performing any graph search during
 * construction. It partitions the dataset into small overlapping leaves via flattened
 * randomized ball carving, computes a bi-directed 2-NN graph inside each leaf, streams the
 * resulting candidate edges into a bounded per-point reservoir (HashPrune), and sparsifies
 * the resulting reservoirs with DirectEdgePrune.
 *
 * This is a prototype: scoring runs in fp16 with fp32 accumulation, and a single partition round
 * is performed (no replication). It ships under `cuvs::neighbors::experimental` for its first
 * release, so the parameter set and the entry points may change.
 *
 * Metrics supported:
 * - L2Expanded
 * - CosineExpanded (note: the defaults are tuned for L2Expanded and may not be optimal here)
 *
 * Requires a GPU of compute capability 8.0 (Ampere) or newer.
 */
struct index_params : cuvs::neighbors::index_params {
  /** Degree of the output graph. */
  uint32_t graph_degree = 64;
  /**
   * Capacity of the per-point HashPrune reservoir. Must be >= graph_degree.
   *
   * Setting this equal to `graph_degree` disables DirectEdgePrune. The default of
   * 2 * graph_degree is what the paper uses and is empirically good.
   */
  uint32_t max_candidates = 128;
  /**
   * Neighbours kept per member of each leaf self-join. At most 32.
   *
   * This is what bounds the candidate supply, and therefore whether the reservoir has more
   * members than the output has slots. Peak memory during the leaf phase is linear in it.
   *
   * The paper uses 2, but 4 was better across the sweeps and benchmarks we tested.
   */
  uint32_t leaf_degree = 4;
  /**
   * Number of LSH hyperplanes used for HashPrune residual hashing. At most 16.
   *
   * More bits mean finer direction buckets and therefore less pruning. The paper uses 12, but we
   * see recall still rising at 14 for a minimal build-time cost.
   */
  uint32_t hash_bits = 14;
  /** Target maximum leaf size. Leaves are guaranteed not to exceed this. */
  uint32_t max_leaf_size = 1024;
  /** Leaves smaller than this are merged with adjacent leaves from the same parent bucket. */
  uint32_t min_leaf_size = 64;
  /** Fraction of a subproblem sampled as leaders, capped by `max_leaders`. */
  float leader_fraction = 0.02f;
  /** Hard cap on the number of leaders sampled per subproblem. */
  uint32_t max_leaders = 1000;
  /** Number of nearest leaders each point is assigned to at the top level. At most 32. */
  uint32_t fanout_level0 = 8;
  /** Number of nearest sub-leaders each point is assigned to at the second level. At most 32. */
  uint32_t fanout_level1 = 4;
  /**
   * Maximum number of levels of the partition tree (assign-then-sort rounds). The first two
   * create overlap from the fanout values; further rounds subdivide still-oversized buckets
   * with fanout 1.
   */
  uint32_t max_rounds = 4;
  /** Alpha parameter of the DirectEdgePrune occlusion rule. Larger retains more edges. */
  float alpha = 1.2f;
  /**
   * Device memory, in MiB, the transient candidate edge-record buffer may occupy. 0 selects a
   * size proportional to the dataset.
   *
   * This dictates how many leaves are processed per batch. Affects only peak memory and build
   * time: a smaller window costs more passes over the reservoirs, a larger one reserves more
   * memory. The graph is the same either way.
   */
  uint32_t leaf_batch_mib = 0;
  /** Seed for leader sampling and LSH hyperplane generation. */
  uint64_t seed = 137ULL;
};

/**
 * @}
 */

static_assert(std::is_aggregate_v<index_params>);

/**
 * @defgroup gpipnn_cpp_build GPiPNN graph build
 * @{
 */

/**
 * @brief Whether this build of the library can run GPiPNN on the current device.
 *
 * False when the device predates compute capability 8.0, and also when the device is new enough
 * but libcuvs was compiled without an sm_80-or-newer image for the leaf join kernel.
 */
auto is_supported() -> bool;

/**
 * @brief Build a GPiPNN proximity graph over `dataset`.
 *
 * The output graph is a fully populated `[n_rows, graph_degree]` adjacency matrix that can be
 * handed directly to a CAGRA index for search:
 *
 * @code{.cpp}
 *   namespace gpipnn = cuvs::neighbors::experimental::gpipnn;
 *
 *   gpipnn::index_params params;
 *   auto graph = raft::make_device_matrix<uint32_t, int64_t>(res, n_rows, params.graph_degree);
 *   gpipnn::build_graph(res, params, dataset, graph.view());
 *
 *   namespace cagra = cuvs::neighbors::cagra;
 *   cagra::index<float, uint32_t> index(res,
 *                                      cuvs::distance::DistanceType::L2Expanded,
 *                                      dataset,
 *                                      raft::make_const_mdspan(graph.view()));
 * @endcode
 *
 * @param[in] res raft resources
 * @param[in] params build parameters
 * @param[in] dataset row-major device matrix [n_rows, dim]; `dim` must be even
 * @param[out] graph row-major device matrix [n_rows, params.graph_degree]
 */
void build_graph(raft::resources const& res,
                 const index_params& params,
                 raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
                 raft::device_matrix_view<uint32_t, int64_t, raft::row_major> graph);

/**
 * @brief Build a GPiPNN proximity graph over a half-precision `dataset`.
 *
 * Identical to the float overload, except the caller's buffer is used directly as the scoring
 * operand, avoiding the fp16 conversion copy.
 *
 * @param[in] res raft resources
 * @param[in] params build parameters
 * @param[in] dataset row-major device matrix [n_rows, dim]; `dim` must be even
 * @param[out] graph row-major device matrix [n_rows, params.graph_degree]
 */
void build_graph(raft::resources const& res,
                 const index_params& params,
                 raft::device_matrix_view<const half, int64_t, raft::row_major> dataset,
                 raft::device_matrix_view<uint32_t, int64_t, raft::row_major> graph);

/**
 * @brief Build a GPiPNN proximity graph over an int8 `dataset`.
 *
 * Identical to the float overload: converts to fp16 for scoring. The conversion is exact, as half
 * represents every integer up to 2048.
 *
 * Note that the scoring copy is `2 * n_rows * dim` bytes whatever the input width, so int8 reduces
 * the footprint of the dataset, not of the build.
 *
 * @param[in] res raft resources
 * @param[in] params build parameters
 * @param[in] dataset row-major device matrix [n_rows, dim]; `dim` must be even
 * @param[out] graph row-major device matrix [n_rows, params.graph_degree]
 */
void build_graph(raft::resources const& res,
                 const index_params& params,
                 raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
                 raft::device_matrix_view<uint32_t, int64_t, raft::row_major> graph);

/**
 * @brief Build a GPiPNN proximity graph over a uint8 `dataset`.
 *
 * Identical to the int8 overload: converts to fp16 for scoring.
 *
 * @param[in] res raft resources
 * @param[in] params build parameters
 * @param[in] dataset row-major device matrix [n_rows, dim]; `dim` must be even
 * @param[out] graph row-major device matrix [n_rows, params.graph_degree]
 */
void build_graph(raft::resources const& res,
                 const index_params& params,
                 raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
                 raft::device_matrix_view<uint32_t, int64_t, raft::row_major> graph);

/**
 * @}
 */

}  // namespace gpipnn
}  // namespace experimental
}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs
namespace CUVS_EXPORT cuvs {
namespace neighbors {
namespace graph_build_params {
/** Stays here beside CAGRA's other graph-build parameter types, which are not in `experimental`. */
using gpipnn_params = cuvs::neighbors::experimental::gpipnn::index_params;
}  // namespace graph_build_params
}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs

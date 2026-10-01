/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#if defined(CUVS_ENABLE_FLOWANN_SEARCH) && defined(CUVS_ENABLE_FLOWANN_BUILD)

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/flowann.hpp>

#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace cuvs::neighbors::cagra::experimental::flowann {

/** Parameters for converting a CAGRA graph into FlowANN's tiered graph representation. */
struct build_params {
  /** Number of consecutive nodes sharing one packed inner-graph row. */
  std::uint16_t node_per_cacheline = 2;
  /** Number of bits used for a node offset inside its partition. */
  std::uint16_t n_bits = 24;
  /** Packed row size in bytes. Zero selects the original average-local-degree heuristic. */
  std::uint32_t inner_graph_row_bytes = 0;
  /** Verify every encoded inner edge and every cross edge before returning. */
  bool validate = true;
};

/** Parameters controlling the locality groups used by the integrated FlowANN build. */
struct grouping_params {
  /** Build and store a grouped graph layout. Disable this to retain the legacy rank-only layout. */
  bool enabled = true;
  /** Final group count. Zero uses one group if IDs fit, otherwise ceil(N * (1 + tolerance) / 2^b).
   * Each k-means node has at most 16 children; this does not limit the final group count.
   */
  std::uint32_t n_groups = 0;
  /** Local-ID width. Zero uses min(24, max(4, align_up_4(ceil(log2(N))))).
   * Explicit widths in [1, 32] remain available for fixed-layout builds.
   */
  std::uint16_t n_bits = 0;
  /** Allowed relative deviation from the average group size, in the open interval `(0, 1)`. */
  double balance_tolerance = 0.10;
  /** Maximum sampled rows per k-means node. Zero shares a 160,000-row budget by subtree
   * leaf count, capped at 10,000 rows per child and floored at 256 rows per child.
   * Sampling is always capped at the actual node size.
   */
  std::size_t training_rows = 0;
  /** Number of rows assigned to groups per GPU batch. Zero targets a 256 MiB vector buffer;
   * distance and label buffers are additional. Does not limit host capacity-repair storage.
   */
  std::size_t assignment_batch_rows = 0;
  /** Number of balanced k-means training iterations. */
  std::uint32_t kmeans_n_iters = 20;
  /** Verify every encoded edge after rearrangement. Intended for tests and small builds. */
  bool validate = false;
};

/** Parameters for the integrated rank/budget FlowANN build. */
struct index_params {
  /** Parameters used to build the complete CAGRA graph before tiering. */
  cuvs::neighbors::cagra::index_params cagra_params{};
  /** Maximum bytes occupied by the final packed resident graph. Grouped builds derive uniform
   * row capacity from this budget, capped at the full graph degree. Zero stores all edges on host.
   */
  std::size_t device_graph_budget_bytes = 0;
  /** Number of consecutive nodes sharing one packed inner-graph row. */
  std::uint16_t node_per_cacheline = 2;
  /** Locality grouping and group-local ID compression parameters. */
  grouping_params grouping{};
  /** Number of medoid seeds generated during build. Zero disables seed generation. */
  std::uint32_t num_seeds = 0;
  /** Number of sampled rows used to train seed centroids. Zero selects an automatic limit. */
  std::size_t seed_training_rows = 0;
  /** Deterministic seed used to rotate the uniform k-means training sample. */
  std::uint64_t seed = 0x9e3779b97f4a7c15ULL;
};

/** Grouping output, reusable when the complete CAGRA graph is already available. */
struct grouping_result {
  /** One group ID in [0, n_groups) per original dataset row; owns its host storage. */
  raft::host_vector<std::uint32_t, int64_t> labels;
  /** Resolved uniform local-ID width, including automatic width selection. */
  std::uint16_t n_bits;
  /** Resolved final group count. */
  std::uint32_t n_groups;
};

/**
 * @brief Assign locality groups through a predetermined hierarchy with capacity repair at each
 * level.
 *
 * Grouping uses vector distances, not graph adjacency. Only the graph shape is inspected; this
 * function does not build or change edges. K-means nodes have at most 16 children. Assignment is
 * batched on the GPU, but capacity repair retains count * children float costs in host memory for
 * the current node. Large inputs therefore still require substantial host memory.
 * This operation synchronizes the resource stream before returning completed host labels.
 *
 * @param res Allocation, stream ordering, and GPU execution resources.
 * @param params Grouping settings, sampling seed, and cagra_params.metric. Supported metrics are
 * L2Expanded, InnerProduct, and CosineExpanded. Other index settings, including the resident graph
 * budget and grouping.enabled, do not affect this standalone operation.
 * @param graph Host graph with original dataset row IDs, N rows, and degree in [1, 255].
 * @param dataset Full-precision host vectors with N rows and positive dimension. N must be positive
 * and fit uint32 graph IDs. Training iterations must be positive and computed distances finite.
 * @return Owning host labels plus the resolved uniform ID width and group count. At most 65,536
 * groups are supported by the graph metadata. A single group bypasses k-means training.
 */
#define CUVS_FLOWANN_DECLARE_GROUP_GRAPH(T)                                      \
  CUVS_EXPORT auto group_graph(                                                  \
    raft::resources const& res,                                                  \
    index_params const& params,                                                  \
    raft::host_matrix_view<const std::uint32_t, int64_t, raft::row_major> graph, \
    cuvs::neighbors::host_standard_dataset_view<T, int64_t> const& dataset) -> grouping_result

CUVS_FLOWANN_DECLARE_GROUP_GRAPH(float);
CUVS_FLOWANN_DECLARE_GROUP_GRAPH(half);
CUVS_FLOWANN_DECLARE_GROUP_GRAPH(std::int8_t);
CUVS_FLOWANN_DECLARE_GROUP_GRAPH(std::uint8_t);
#undef CUVS_FLOWANN_DECLARE_GROUP_GRAPH

/**
 * @brief Build a FlowANN index by constructing and immediately tiering a final CAGRA graph.
 *
 * The device-padded overload copies the dense dataset into its owning output bundle. The host
 * overload ranks with the full-precision input and produces an owning VPQ-F16 bundle using
 * `vpq_params`. Both overloads retain every final CAGRA edge. Group-local edges selected by the
 * packed-row budget are stored on the device, and all remaining edges are stored in the host cross
 * graph. ACE graph build parameters are not supported.
 *
 * `res` provides allocation, stream ordering, and GPU execution resources. `params` controls CAGRA
 * construction, balanced locality grouping, resident-graph budgeting, packing, and optional seeds.
 * Grouped builds rank graph edges with the full-precision input before choosing resident edges,
 * reorder graph and dataset rows together, and expose original row IDs through `source_indices()`.
 * `dataset` is the full-precision input and its row count must fit uint32 graph IDs. The host
 * overload additionally accepts `vpq_params` for the output dataset. Each overload returns an
 * owning dataset/index bundle whose index view remains valid for the bundle lifetime.
 */
#define CUVS_FLOWANN_DECLARE_INTEGRATED_BUILD(T)                                                 \
  CUVS_EXPORT auto build(raft::resources const& res,                                             \
                         index_params const& params,                                             \
                         cuvs::neighbors::device_padded_dataset_view<T, int64_t> const& dataset) \
    -> device_padded_index_bundle<T, std::uint32_t>;                                             \
  CUVS_EXPORT auto build(raft::resources const& res,                                             \
                         index_params const& params,                                             \
                         cuvs::neighbors::vpq_params const& vpq_params,                          \
                         cuvs::neighbors::host_standard_dataset_view<T, int64_t> const& dataset) \
    -> vpq_f16_index_bundle<T, std::uint32_t>

CUVS_FLOWANN_DECLARE_INTEGRATED_BUILD(float);
CUVS_FLOWANN_DECLARE_INTEGRATED_BUILD(half);
CUVS_FLOWANN_DECLARE_INTEGRATED_BUILD(std::int8_t);
CUVS_FLOWANN_DECLARE_INTEGRATED_BUILD(std::uint8_t);

#undef CUVS_FLOWANN_DECLARE_INTEGRATED_BUILD

/**
 * Convert a CAGRA index into an owning FlowANN index. `partition_labels` contains one partition
 * identifier per CAGRA node. The rvalue VPQ overload releases the full CAGRA graph before
 * allocating the FlowANN graph to bound peak device memory use.
 */
#define CUVS_FLOWANN_DECLARE_BUILD(T)                                                         \
  CUVS_EXPORT auto build_from_cagra(                                                          \
    raft::resources const& res,                                                               \
    build_params const& params,                                                               \
    cuvs::neighbors::cagra::device_padded_index<T, std::uint32_t> const& cagra_index,         \
    raft::host_vector_view<const std::uint32_t, int64_t> partition_labels,                    \
    std::optional<raft::host_vector_view<const std::uint32_t, int64_t>> seeds = std::nullopt) \
    -> device_padded_index_bundle<T, std::uint32_t>;                                          \
  CUVS_EXPORT auto build_from_cagra(                                                          \
    raft::resources const& res,                                                               \
    build_params const& params,                                                               \
    cuvs::neighbors::cagra::device_pq_index<T, std::uint32_t> const& cagra_index,             \
    raft::host_vector_view<const std::uint32_t, int64_t> partition_labels,                    \
    std::optional<raft::host_vector_view<const std::uint32_t, int64_t>> seeds = std::nullopt) \
    -> vpq_f16_index_bundle<T, std::uint32_t>;                                                \
  CUVS_EXPORT auto build_from_cagra(                                                          \
    raft::resources const& res,                                                               \
    build_params const& params,                                                               \
    cuvs::neighbors::cagra::device_pq_index<T, std::uint32_t>&& cagra_index,                  \
    std::unique_ptr<cuvs::neighbors::device_vpq_dataset<half, int64_t>>&& vpq_dataset,        \
    raft::host_vector_view<const std::uint32_t, int64_t> partition_labels,                    \
    std::optional<raft::host_vector_view<const std::uint32_t, int64_t>> seeds = std::nullopt) \
    -> vpq_f16_index_bundle<T, std::uint32_t>

CUVS_FLOWANN_DECLARE_BUILD(float);
CUVS_FLOWANN_DECLARE_BUILD(half);
CUVS_FLOWANN_DECLARE_BUILD(std::int8_t);
CUVS_FLOWANN_DECLARE_BUILD(std::uint8_t);

#undef CUVS_FLOWANN_DECLARE_BUILD

}  // namespace cuvs::neighbors::cagra::experimental::flowann

#endif  // CUVS_ENABLE_FLOWANN_SEARCH && CUVS_ENABLE_FLOWANN_BUILD

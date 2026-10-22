/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common.cuh"
#include "direct_edge_prune.cuh"
#include "hash_prune.cuh"
#include "leaf.cuh"
#include "partition.cuh"
#include "segmented_topk.cuh"

#include <cuvs/neighbors/gpipnn.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <type_traits>

namespace cuvs::neighbors::experimental::gpipnn::detail {

/**
 * Record-buffer budget per row when `index_params::leaf_batch_mib` is left on auto.
 *
 * Scaling the budget with `n_rows` rather than fixing it keeps peak memory linear in the
 * dataset size.
 */
inline constexpr int64_t kAutoLeafBatchBytesPerRow = 256;

/**
 * Floor and ceiling on the auto budget: one batch for small builds, bounded reservation.
 *
 * The floor binds only below 2M rows and can cost at most itself in extra peak.
 *
 * Nothing here can change the graph: the batch division does not enter the answer (see
 * `hash_prune_merge_kernel`), verified by checksum across a 1000x range of budgets.
 */
inline constexpr int64_t kMinLeafBatchBytes = int64_t{512} << 20;
inline constexpr int64_t kMaxLeafBatchBytes = int64_t{6} << 30;

/** Bytes the transient edge-record buffer is allowed to occupy at any one time. */
inline auto leaf_batch_bytes(const index_params& params, int64_t n_rows) -> int64_t
{
  if (params.leaf_batch_mib > 0) { return static_cast<int64_t>(params.leaf_batch_mib) << 20; }
  return std::clamp(n_rows * kAutoLeafBatchBytesPerRow, kMinLeafBatchBytes, kMaxLeafBatchBytes);
}

/**
 * Peak record-array bytes one leaf member contributes.
 *
 * A member emits `leaf_degree` selected neighbours, each of which becomes a forward and a
 * reverse record of `sizeof(key) + sizeof(target)` bytes, and the radix sort needs a second
 * buffer of the same size. `leaf_pick`'s own selection outputs are smaller than the sort buffer
 * they are freed before, so they never set the peak.
 */
inline auto record_bytes_per_member(int64_t leaf_degree) -> int64_t
{
  return leaf_degree * 2 * static_cast<int64_t>(sizeof(uint64_t) + sizeof(id_t)) * 2;
}

/**
 * Builds a GPiPNN proximity graph.
 *
 * The four phases are strictly sequential, but the third does not wait for the second to finish:
 * the candidate edge records are transient -- HashPrune folds them into a fixed-size reservoir and
 * nothing downstream reads them again -- so the leaf join runs a range of leaves at a time and
 * each range's records are merged and released before the next is built. Peak memory is then the
 * long-lived arrays plus a budget independent of how many leaves each point belongs to. Cutting
 * the work into ranges does not change the answer; see `hash_prune_merge_kernel`.
 *
 * See cuvs::neighbors::experimental::gpipnn::build_graph.
 */
template <typename DataT>
void build_graph(raft::resources const& res,
                 const index_params& params,
                 raft::device_matrix_view<const DataT, int64_t, raft::row_major> dataset,
                 raft::device_matrix_view<id_t, int64_t, raft::row_major> graph)
{
  static_assert(std::is_same_v<DataT, float> || std::is_same_v<DataT, half> ||
                  std::is_same_v<DataT, int8_t> || std::is_same_v<DataT, uint8_t>,
                "gpipnn: only float, half, int8 and uint8 datasets are supported");

  // Before anything is validated or uploaded: on a device this cannot run on, the leaf join is a
  // stub that would return an empty graph rather than fail. Callers with a fallback should ask
  // `gpipnn_arch_supported()` instead of relying on this.
  check_gpipnn_arch();

  const int64_t n_rows = dataset.extent(0);
  const int64_t dim    = dataset.extent(1);
  check_params(params, n_rows, dim);
  RAFT_EXPECTS(graph.extent(0) == n_rows,
               "gpipnn: graph must have the same number of rows as the dataset");
  RAFT_EXPECTS(graph.extent(1) == static_cast<int64_t>(params.graph_degree),
               "gpipnn: graph must have exactly params.graph_degree columns");
  RAFT_EXPECTS(n_rows > static_cast<int64_t>(params.graph_degree),
               "gpipnn: dataset must have more rows (%zu) than graph_degree (%u)",
               static_cast<size_t>(n_rows),
               params.graph_degree);
  RAFT_EXPECTS(n_rows < static_cast<int64_t>(kInvalidId),
               "gpipnn: dataset is too large to address with 32-bit ids");

  auto stream = raft::resource::get_cuda_stream(res).get();

  // All scoring runs on half operands with fp32 accumulation, mirroring the paper's FP16 mode. A
  // half caller already holds the right buffer, so that path allocates nothing; every other type
  // is converted here and only here, which is why the type disappears from the rest of this
  // function. The copy is `2 * n_rows * dim` bytes whatever the input width, so int8 reduces the
  // caller's footprint, not the build's.
  //
  // Cosine is also handled entirely here, by scaling each row to unit norm as it is converted (see
  // `normalize_rows_to_half_kernel`). That costs a half caller the zero-copy path, since its view
  // is const and cannot be normalized in place.
  const bool normalize = params.metric == cuvs::distance::DistanceType::CosineExpanded;
  rmm::device_uvector<half> half_storage(0, stream);
  const half* scoring_data = nullptr;
  if constexpr (std::is_same_v<DataT, half>) {
    if (!normalize) { scoring_data = dataset.data_handle(); }
  }
  if (scoring_data == nullptr) {
    half_storage.resize(n_rows * dim, stream);
    if (normalize) {
      normalize_rows_to_half_kernel<<<n_rows, 256, 0, stream>>>(
        dataset.data_handle(), half_storage.data(), n_rows, dim);
    } else {
      const int64_t threads = 256;
      const int64_t blocks  = (n_rows * dim + threads - 1) / threads;
      cast_to_half_kernel<<<blocks, threads, 0, stream>>>(
        dataset.data_handle(), half_storage.data(), n_rows * dim);
    }
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    scoring_data = half_storage.data();
  }
  auto scoring_view =
    raft::make_device_matrix_view<const half, int64_t, raft::row_major>(scoring_data, n_rows, dim);

  rmm::device_uvector<float> norms(n_rows, stream);
  row_norms_kernel<<<n_rows, 256, 0, stream>>>(scoring_data, n_rows, dim, norms.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  auto norms_view = raft::make_device_vector_view<const float, int64_t>(norms.data(), n_rows);

  carve_stats stats;
  auto leaves = ball_carve(res, scoring_view, norms_view, params, stats);
  RAFT_LOG_DEBUG(
    "gpipnn: %ld leaves over %ld memberships after %ld rounds (%ld chopped, %ld merged, %ld still "
    "oversized)",
    leaves.count(),
    leaves.total(),
    stats.rounds_run,
    stats.chopped_runs,
    stats.merged_runs,
    stats.oversized_left);

  // The paper picks top-2, the smallest choice that still yields a connected candidate set once
  // every edge is bi-directed. That is also too few to fill a degree-64 reservoir on real data
  // once duplicate targets and the LSH dedup have taken their share, which starves the prune.
  const auto leaf_degree = static_cast<int64_t>(params.leaf_degree);
  const auto hash_bits   = static_cast<int32_t>(params.hash_bits);

  auto sketches = compute_sketches(res, scoring_view, hash_bits, params.seed + 7);
  auto sketches_view =
    raft::make_device_vector_view<const float, int64_t>(sketches.data(), n_rows * hash_bits);
  auto acc = make_accumulator(res, n_rows, params.max_candidates);
  rmm::device_uvector<int64_t> row_begin(n_rows + 1, stream);

  // Leaves are packed into ranges by member count, since that is what the record array is
  // proportional to. A single leaf always forms a range on its own if it has to, so the loop
  // makes progress whatever the budget.
  const int64_t per_member  = record_bytes_per_member(leaf_degree);
  const int64_t max_members = std::max<int64_t>(1, leaf_batch_bytes(params, n_rows) / per_member);

  int64_t n_batches = 0;
  int64_t n_records = 0;
  for (int64_t lo = 0; lo < leaves.count();) {
    int64_t hi      = lo;
    int64_t members = 0;
    while (hi < leaves.count() && (hi == lo || members + leaves.size_of(hi) <= max_members)) {
      members += leaves.size_of(hi);
      hi++;
    }
    auto records = leaf_pick_range(res, scoring_view, norms_view, leaves, leaf_degree, lo, hi);
    n_records += records.n_valid;
    hash_prune_merge(res, sketches_view, records, params, row_begin, acc);
    lo = hi;
    n_batches++;
  }
  RAFT_LOG_DEBUG("gpipnn: %ld candidate edge records over %ld leaf batches (<= %ld members each)",
                 n_records,
                 n_batches,
                 max_members);

  // The prune's row order is the last thing the partition is needed for: it is the leaf grouping
  // reduced to `n_rows` ids, which is 45 MB at 11.16M against the 1.4 GiB the memberships hold.
  // See `build_prune_order` for what it buys and why it cannot change the graph.
  auto prune_order = build_prune_order(res, leaves, n_rows);

  // Nothing below reads the partition, the sketches, the owner bounds or the per-entry hashes.
  leaves.members.resize(0, stream);
  leaves.members.shrink_to_fit(stream);
  sketches.resize(0, stream);
  sketches.shrink_to_fit(stream);
  row_begin.resize(0, stream);
  row_begin.shrink_to_fit(stream);
  acc.release_hashes(stream);

  direct_edge_prune(
    res,
    acc,
    params,
    graph,
    static_cast<int64_t>(prune_order.size()) == n_rows ? prune_order.data() : nullptr);
}

}  // namespace cuvs::neighbors::experimental::gpipnn::detail

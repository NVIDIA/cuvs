/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../../core/mnmg_comms.cuh"
#include "kmeans.cuh"
#include "kmeans_common.cuh"
#include "kmeans_mg_batched_init.cuh"

#include "../kmeans.cuh"

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/distance/distance.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/matrix/init.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

namespace cuvs::cluster::kmeans::mg::detail {

/**
 * @brief Compute the global cluster cost (psi) over partitioned device data.
 *
 * For every non-empty partition, computes the per-row minimum distance to the
 * current candidate centroid set, reduces locally with `cluster_cost`, then
 * allreduces across ranks. Returns the host-side global cost.
 *
 * Shared scratch (`L2NormBuf_OR_DistBuf`, `workspace`) is passed in by
 * reference so it can be reused across repeated calls in the kmeans||
 * candidate-sampling loop.
 *
 * @tparam DataT  float or double
 * @tparam IndexT int or int64_t
 *
 * @param handle               RAFT resources for this rank.
 * @param params               KMeans parameters (uses metric and batch sizes).
 * @param X_parts              Local device matrix partitions on this rank.
 * @param part_offsets         Prefix-sum of partition row counts (size
 *                             `X_parts.size() + 1`).
 * @param n_local              Total local rows (`part_offsets.back()`).
 * @param L2NormX              Precomputed L2 norms of every local row (length
 *                             `>= n_local`). Sliced per partition.
 * @param potentialCentroids   Current candidate centroid set (the "C" buffer).
 * @param minClusterDistance   Per-row min-distance scratch (length `>= n_local`).
 * @param L2NormBuf_OR_DistBuf Scratch buffer reused across `min_cluster_distance`
 *                             calls; grown as needed.
 * @param workspace            General scratch buffer.
 * @param comms                mnmg_comms wrapper for the cross-rank allreduce.
 * @return Global cluster cost (psi).
 */
template <typename DataT, typename IndexT>
DataT compute_global_cluster_cost(
  const raft::resources& handle,
  const cuvs::cluster::kmeans::params& params,
  const std::vector<raft::device_matrix_view<const DataT, IndexT>>& X_parts,
  const std::vector<IndexT>& part_offsets,
  IndexT n_local,
  raft::device_vector_view<DataT, IndexT> L2NormX,
  raft::device_matrix_view<DataT, IndexT> potentialCentroids,
  raft::device_vector_view<DataT, IndexT> minClusterDistance,
  rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,
  rmm::device_uvector<char>& workspace,
  const mnmg_comms& comms)
{
  cudaStream_t stream   = comms.stream();
  const auto n_features = static_cast<IndexT>(potentialCentroids.extent(1));

  auto d_partial = raft::make_device_scalar<DataT>(handle, DataT{0});

  for (std::size_t p = 0; p < X_parts.size(); ++p) {
    auto part_rows = static_cast<IndexT>(X_parts[p].extent(0));
    if (part_rows == 0) { continue; }
    auto x_slice = raft::make_device_matrix_view<const DataT, IndexT>(
      X_parts[p].data_handle(), part_rows, n_features);
    auto mcd_slice = raft::make_device_vector_view<DataT, IndexT>(
      minClusterDistance.data_handle() + part_offsets[p], part_rows);
    auto norm_slice = raft::make_device_vector_view<DataT, IndexT>(
      L2NormX.data_handle() + part_offsets[p], part_rows);

    cuvs::cluster::kmeans::min_cluster_distance<DataT, IndexT>(handle,
                                                               x_slice,
                                                               potentialCentroids,
                                                               mcd_slice,
                                                               norm_slice,
                                                               L2NormBuf_OR_DistBuf,
                                                               params.metric,
                                                               params.batch_samples,
                                                               params.batch_centroids,
                                                               workspace);
  }

  if (n_local > 0) {
    auto mcd_view =
      raft::make_device_vector_view<DataT, IndexT>(minClusterDistance.data_handle(), n_local);
    cuvs::cluster::kmeans::cluster_cost<DataT, IndexT>(
      handle, mcd_view, workspace, d_partial.view(), raft::add_op{});
  }

  comms.allreduce(d_partial.data_handle(), d_partial.data_handle(), 1);

  DataT psi_h = DataT{0};
  raft::copy(&psi_h, d_partial.data_handle(), 1, stream);
  raft::resource::sync_stream(handle);
  return psi_h;
}

/**
 * @brief Truly distributed scalable KMeans++ initialization for the device path.
 *
 * Implements the multi-GPU variant of the scalable KMeans++ algorithm
 * (Bahmani et al., 2012) with no centralized sampling on root. Each rank owns
 * one or more local device matrix partitions (`X_parts`). The algorithm
 * distributes cluster-cost computation and candidate sampling across ranks,
 * exchanging candidate sets with `allgatherv`, and reclustering the merged
 * weighted candidate set on every rank with the same RNG seed (so the result
 * is deterministic across ranks).
 *
 * @tparam DataT  float or double
 * @tparam IndexT int or int64_t
 *
 * @param handle             RAFT resources for this rank (the per-rank
 *                           sub-handle when using SNMG/NCCL).
 * @param params             KMeans parameters; only L2Expanded /
 *                           L2SqrtExpanded metrics are supported.
 * @param X_parts            Local device matrix partitions for this rank.
 * @param n_features         Feature dimension (must match every partition and
 *                           the output centroids).
 * @param centroidsRawData   [out] Initial centroids on this rank
 *                           [n_clusters x n_features].
 * @param workspace          Resizable scratch buffer shared with the helpers.
 * @param rank_counts        Global row counts indexed by rank.
 * @param global_n           Sum of `rank_counts`.
 * @param rank               This rank's id.
 * @param num_ranks          Total number of ranks.
 * @param comms              mnmg_comms wrapper (NCCL or raft comms backend).
 */
template <typename DataT, typename IndexT>
void initKMeansPlusPlus_distributed(
  const raft::resources& handle,
  const cuvs::cluster::kmeans::params& params,
  const std::vector<raft::device_matrix_view<const DataT, IndexT>>& X_parts,
  IndexT n_features,
  raft::device_matrix_view<DataT, IndexT> centroidsRawData,
  rmm::device_uvector<char>& workspace,
  const std::vector<IndexT>& rank_counts,
  IndexT global_n,
  int rank,
  int num_ranks,
  const mnmg_comms& comms)
{
  using cuvs::cluster::kmeans::detail::SamplingOp;

  cudaStream_t stream   = comms.stream();
  const auto n_clusters = static_cast<IndexT>(params.n_clusters);
  const auto metric     = params.metric;

  // Clamp oversampling factor to 1.0 to maintain the distributed initialization with minimum
  // selection probability guided by the cost alone.
  const double effective_oversampling_factor =
    params.oversampling_factor < double{1} ? double{1} : params.oversampling_factor;

  RAFT_EXPECTS(metric == cuvs::distance::DistanceType::L2Expanded ||
                 metric == cuvs::distance::DistanceType::L2SqrtExpanded,
               "Distributed KMeans++ init only supports L2Expanded or L2SqrtExpanded metrics");
  RAFT_EXPECTS(num_ranks > 0, "num_ranks must be positive");
  RAFT_EXPECTS(global_n >= n_clusters,
               "global row count (%zu) must be >= n_clusters (%zu)",
               static_cast<size_t>(global_n),
               static_cast<size_t>(n_clusters));

  IndexT n_local = 0;
  std::vector<IndexT> part_offsets;
  part_offsets.reserve(X_parts.size() + 1);
  part_offsets.push_back(IndexT{0});
  for (auto const& X_part : X_parts) {
    RAFT_EXPECTS(static_cast<IndexT>(X_part.extent(1)) == n_features,
                 "all partitions must share the centroid feature dimension");
    n_local += static_cast<IndexT>(X_part.extent(0));
    part_offsets.push_back(n_local);
  }

  raft::random::RngState rng(params.rng_state.seed, raft::random::GeneratorType::GenPhilox);

  auto d_rp = raft::make_device_scalar<int>(handle, 0);
  int rp    = 0;

  // Step 1.1 - choose the source rank.
  if (rank == CUVS_KMEANS_COMM_ROOT) {
    // Only draw from ranks that actually own at least one row.
    std::vector<int> non_empty_ranks;
    non_empty_ranks.reserve(static_cast<std::size_t>(num_ranks));
    for (int r = 0; r < num_ranks; ++r) {
      if (rank_counts[r] > IndexT{0}) { non_empty_ranks.push_back(r); }
    }
    RAFT_EXPECTS(!non_empty_ranks.empty(),
                 "all ranks have zero local rows; cannot pick a source rank for KMeans++ init");
    std::mt19937 gen(params.rng_state.seed);
    std::uniform_int_distribution<std::size_t> rank_dist(std::size_t{0},
                                                         non_empty_ranks.size() - 1);
    rp = non_empty_ranks[rank_dist(gen)];
    raft::copy(d_rp.data_handle(), &rp, 1, stream);
  }
  comms.bcast(d_rp.data_handle(), static_cast<size_t>(1), CUVS_KMEANS_COMM_ROOT);
  raft::copy(&rp, d_rp.data_handle(), 1, stream);
  raft::resource::sync_stream(handle);

  // Step 1.2 - the source rank picks one of its local rows uniformly.
  // Matches the original: re-seed mt19937(seed) so cIdx is the first draw of a
  // fresh generator (not the second draw of the same one used for rp).
  IndexT chosen_local_idx = -1;
  // Growable buffer of candidate centroids (the "C" set).
  rmm::device_uvector<DataT> centroidsBuf(static_cast<std::size_t>(n_features), stream);
  auto potentialCentroids =
    raft::make_device_matrix_view<DataT, IndexT>(centroidsBuf.data(), IndexT{1}, n_features);
  DataT* initialCentroid = centroidsBuf.data();
  if (rank == rp) {
    RAFT_EXPECTS(n_local > 0,
                 "selected source rank %d has no local rows; cannot pick an initial centroid",
                 rp);
    std::mt19937 row_gen(params.rng_state.seed);
    std::uniform_int_distribution<IndexT> row_dist(IndexT{0}, n_local - 1);
    chosen_local_idx             = row_dist(row_gen);
    auto [part_idx, row_in_part] = locate_local_row(part_offsets, chosen_local_idx);
    initialCentroid =
      const_cast<DataT*>(X_parts[part_idx].data_handle() + row_in_part * n_features);
  }
  // Step 1.3 - broadcast the chosen initial centroid to all ranks.
  comms.bcast(initialCentroid, centroidsBuf.data(), static_cast<size_t>(n_features), rp);

  // Per-rank "is this local row already a chosen centroid" bitmap.
  auto isSampleCentroid =
    raft::make_device_vector<std::uint8_t, IndexT>(handle, std::max(n_local, IndexT{1}));
  if (n_local > 0) { raft::matrix::fill(handle, isSampleCentroid.view(), std::uint8_t{0}); }
  if (rank == rp) {
    const std::uint8_t one = 1;
    raft::copy(isSampleCentroid.data_handle() + chosen_local_idx, &one, 1, stream);
  }

  // Per-rank working buffers spanning the rank's local row range.
  auto L2NormX = raft::make_device_vector<DataT, IndexT>(handle, std::max(n_local, IndexT{1}));
  auto minClusterDistance =
    raft::make_device_vector<DataT, IndexT>(handle, std::max(n_local, IndexT{1}));
  auto uniformRands = raft::make_device_vector<DataT, IndexT>(handle, std::max(n_local, IndexT{1}));

  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);

  // Precompute L2 norms for every local row (the input X parts are constant
  // for the duration of init).
  for (std::size_t p = 0; p < X_parts.size(); ++p) {
    auto part_rows = static_cast<IndexT>(X_parts[p].extent(0));
    if (part_rows == 0) { continue; }
    auto x_slice = raft::make_device_matrix_view<const DataT, IndexT>(
      X_parts[p].data_handle(), part_rows, n_features);
    auto norm_slice = raft::make_device_vector_view<DataT, IndexT>(
      L2NormX.data_handle() + part_offsets[p], part_rows);
    raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(handle, x_slice, norm_slice);
  }

  // Step 2: psi <- phi_X(C).
  DataT psi = compute_global_cluster_cost<DataT, IndexT>(handle,
                                                         params,
                                                         X_parts,
                                                         part_offsets,
                                                         n_local,
                                                         L2NormX.view(),
                                                         potentialCentroids,
                                                         minClusterDistance.view(),
                                                         L2NormBuf_OR_DistBuf,
                                                         workspace,
                                                         comms);

  const int niter = std::min(8, static_cast<int>(std::ceil(std::log(psi))));
  RAFT_LOG_DEBUG(
    "Distributed KMeans||: rank=%d, psi=%g, niter=%d", rank, static_cast<double>(psi), niter);

  // Steps 3-6: sample candidates `O(log psi)` times, gathering across ranks.
  for (int iter = 0; iter < niter; ++iter) {
    if (iter > 0)
      psi = compute_global_cluster_cost<DataT, IndexT>(handle,
                                                       params,
                                                       X_parts,
                                                       part_offsets,
                                                       n_local,
                                                       L2NormX.view(),
                                                       potentialCentroids,
                                                       minClusterDistance.view(),
                                                       L2NormBuf_OR_DistBuf,
                                                       workspace,
                                                       comms);

    if (n_local > 0) {
      auto rands_view =
        raft::make_device_vector_view<DataT, IndexT>(uniformRands.data_handle(), n_local);
      raft::random::uniform(handle, rng, rands_view.data_handle(), n_local, DataT{0}, DataT{1});
    }

    // Sample per partition; collect into a rank-local concatenated buffer.
    std::vector<rmm::device_uvector<DataT>> per_part_cp;
    per_part_cp.reserve(X_parts.size());
    IndexT total_local_sampled = 0;

    for (std::size_t p = 0; p < X_parts.size(); ++p) {
      per_part_cp.emplace_back(0, stream);
      auto& inRankCp = per_part_cp.back();

      auto part_rows = static_cast<IndexT>(X_parts[p].extent(0));
      if (part_rows == 0) { continue; }

      auto x_slice = raft::make_device_matrix_view<const DataT, IndexT>(
        X_parts[p].data_handle(), part_rows, n_features);
      auto mcd_slice = raft::make_device_vector_view<DataT, IndexT>(
        minClusterDistance.data_handle() + part_offsets[p], part_rows);
      auto flag_slice = raft::make_device_vector_view<std::uint8_t, IndexT>(
        isSampleCentroid.data_handle() + part_offsets[p], part_rows);

      SamplingOp<DataT, IndexT> select_op(psi,
                                          effective_oversampling_factor,
                                          n_clusters,
                                          uniformRands.data_handle() + part_offsets[p],
                                          flag_slice.data_handle());

      cuvs::cluster::kmeans::sample_centroids<DataT, IndexT>(
        handle, x_slice, mcd_slice, flag_slice, select_op, inRankCp, workspace);
      total_local_sampled += static_cast<IndexT>(inRankCp.size() / n_features);
    }

    // Concatenate this rank's per-part sampled rows into a single contiguous
    // buffer suitable for allgatherv.
    rmm::device_uvector<DataT> local_cp(static_cast<std::size_t>(total_local_sampled) * n_features,
                                        stream);
    std::size_t write_off = 0;
    for (auto const& part_cp : per_part_cp) {
      if (part_cp.size() == 0) { continue; }
      raft::copy(local_cp.data() + write_off, part_cp.data(), part_cp.size(), stream);
      write_off += part_cp.size();
    }

    // Exchange per-rank candidate counts, then concatenate every rank's
    // candidates onto the back of every rank's centroidsBuf.
    auto d_my_count = raft::make_device_scalar<IndexT>(handle, total_local_sampled);
    auto d_counts =
      raft::make_device_vector<IndexT, IndexT>(handle, static_cast<IndexT>(num_ranks));
    comms.allgather(d_my_count.data_handle(), d_counts.data_handle(), 1);
    std::vector<IndexT> h_counts(num_ranks);
    raft::copy(h_counts.data(), d_counts.data_handle(), num_ranks, stream);
    raft::resource::sync_stream(handle);

    std::vector<std::size_t> sizes(num_ranks);
    std::vector<std::size_t> displs(num_ranks);
    std::size_t cumul = 0;
    IndexT total_new  = 0;
    for (int r = 0; r < num_ranks; ++r) {
      sizes[r]  = static_cast<std::size_t>(h_counts[r]) * static_cast<std::size_t>(n_features);
      displs[r] = cumul;
      cumul += sizes[r];
      total_new += h_counts[r];
    }

    if (total_new > 0) {
      auto old_size = centroidsBuf.size();
      centroidsBuf.resize(old_size + static_cast<std::size_t>(total_new) * n_features, stream);
      comms.allgatherv(local_cp.data(),
                       centroidsBuf.end() - total_new * n_features,
                       sizes.data(),
                       displs.data(),
                       local_cp.size());
    }

    IndexT tot_centroids = static_cast<IndexT>(potentialCentroids.extent(0)) + total_new;
    potentialCentroids =
      raft::make_device_matrix_view<DataT, IndexT>(centroidsBuf.data(), tot_centroids, n_features);
  }

  RAFT_LOG_DEBUG("Distributed KMeans||: rank=%d, total candidates = %d",
                 rank,
                 static_cast<int>(potentialCentroids.extent(0)));

  // Step 7+8: reweight the candidates and recluster down to n_clusters.
  if (static_cast<IndexT>(potentialCentroids.extent(0)) > n_clusters) {
    const auto n_candidates = static_cast<IndexT>(potentialCentroids.extent(0));
    auto weight             = raft::make_device_vector<DataT, IndexT>(handle, n_candidates);
    raft::matrix::fill(handle, weight.view(), DataT{0});

    auto part_weight = raft::make_device_vector<DataT, IndexT>(handle, n_candidates);
    for (std::size_t p = 0; p < X_parts.size(); ++p) {
      auto part_rows = static_cast<IndexT>(X_parts[p].extent(0));
      if (part_rows == 0) { continue; }
      auto x_slice = raft::make_device_matrix_view<const DataT, IndexT>(
        X_parts[p].data_handle(), part_rows, n_features);
      auto norm_slice = raft::make_device_vector_view<DataT, IndexT>(
        L2NormX.data_handle() + part_offsets[p], part_rows);

      raft::matrix::fill(handle, part_weight.view(), DataT{0});
      cuvs::cluster::kmeans::count_samples_in_cluster<DataT, IndexT>(
        handle, params, x_slice, norm_slice, potentialCentroids, workspace, part_weight.view());

      raft::linalg::add(handle,
                        raft::make_const_mdspan(weight.view()),
                        raft::make_const_mdspan(part_weight.view()),
                        weight.view());
    }

    comms.allreduce(
      weight.data_handle(), weight.data_handle(), static_cast<std::size_t>(n_candidates));

    // Step 8: deterministic recluster on every rank using the merged weights.
    auto const_centroids = raft::make_device_matrix_view<const DataT, IndexT>(
      potentialCentroids.data_handle(), potentialCentroids.extent(0), n_features);

    cuvs::cluster::kmeans::init_plus_plus<DataT, IndexT>(
      handle, params, const_centroids, centroidsRawData, workspace);

    auto inertia_out = raft::make_host_scalar<DataT>(0);
    auto n_iter_out  = raft::make_host_scalar<IndexT>(0);

    cuvs::cluster::kmeans::params default_params;
    cuvs::cluster::kmeans::params recluster_params = params;
    recluster_params.rng_state                     = default_params.rng_state;
    recluster_params.init   = cuvs::cluster::kmeans::params::InitMethod::Array;
    recluster_params.n_init = 1;

    auto weight_opt = std::make_optional(raft::make_const_mdspan(weight.view()));
    cuvs::cluster::kmeans::fit(handle,
                               recluster_params,
                               raft::make_const_mdspan(potentialCentroids),
                               weight_opt,
                               centroidsRawData,
                               inertia_out.view(),
                               n_iter_out.view());

  } else if (static_cast<IndexT>(potentialCentroids.extent(0)) < n_clusters) {
    const IndexT n_random = n_clusters - static_cast<IndexT>(potentialCentroids.extent(0));
    RAFT_LOG_DEBUG(
      "[r=%d] dist_init: candidates (%d) < n_clusters (%d); globally sampling %d random rows",
      rank,
      static_cast<int>(potentialCentroids.extent(0)),
      static_cast<int>(n_clusters),
      static_cast<int>(n_random));

    auto random_centroids = sample_global_rows(handle,
                                               params,
                                               X_parts,
                                               n_features,
                                               n_random,
                                               rank,
                                               rank_counts,
                                               global_n,
                                               comms,
                                               /*result_on_all_ranks=*/true);

    raft::copy(centroidsRawData.data_handle(),
               random_centroids.data_handle(),
               random_centroids.size(),
               stream);

    // copy centroids generated during kmeans|| iteration to the buffer
    raft::copy(centroidsRawData.data_handle() + static_cast<std::size_t>(n_random) * n_features,
               potentialCentroids.data_handle(),
               potentialCentroids.size(),
               stream);
  } else {
    raft::copy(centroidsRawData.data_handle(),
               potentialCentroids.data_handle(),
               potentialCentroids.size(),
               stream);
  }
}

}  // namespace cuvs::cluster::kmeans::mg::detail

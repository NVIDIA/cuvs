/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../kmeans.cuh"
#include "kmeans.cuh"
#include "kmeans_common.cuh"
#include "kmeans_mg_batched_init.cuh"
#include "kmeans_mg_distributed_init.cuh"

#include "../../core/mnmg_comms.cuh"
#include "../../core/omp_wrapper.hpp"
#include "../../neighbors/detail/ann_utils.cuh"

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/distance/distance.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/kvp.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/pinned_mdarray.hpp>
#include <raft/core/pinned_mdspan.hpp>
#include <raft/core/resource/comms.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/multi_gpu.hpp>
#include <raft/core/resource/nccl_comm.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/matrix/init.cuh>
#include <raft/matrix/sample_rows.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

namespace cuvs::cluster::kmeans::mg::detail {

using cuvs::core::detail::mnmg_comms;

/**
 * @brief Shared multi-GPU k-means fit core, called per rank.
 *
 * Backs both Path 1 (OMP threads sharing a clique with NCCL comms) and Path 2
 * (one rank per process with RAFT comms). The active backend is selected via
 * `raft::resource::is_multi_gpu(handle)`. Each rank owns one or more local
 * partitions, supplied as a vector of mdspans with the same accessor type.
 * Individual partitions may be empty, and an entire rank may have zero local
 * rows; only the global aggregate across all ranks must satisfy
 * `global_n >= n_clusters` for non-Array initialization. The implementation
 * streams every non-empty local partition through Lloyd iterations using
 * batched device-side reductions, allreducing partial centroid sums, weights,
 * convergence state, and clustering cost at the end of each iteration.
 * Best-of-`n_init` is tracked per rank. RAFT comms ranks each write their own caller-provided
 * outputs; SNMG OMP threads share caller-provided outputs, so only rank 0 writes them.
 *
 * @tparam DataT    Data / weight type (float or double)
 * @tparam IndexT   Index type (int or int64_t)
 * @tparam Accessor RAFT mdspan accessor type for the input data and optional
 *                  weights.
 *
 * @param[in]  handle        RAFT resources. For Path 1 this is the shared
 *                           clique; for Path 2 this is the per-process handle.
 * @param[in]  params        K-means parameters (n_clusters, init, max_iter,
 *                           n_init, tol, metric, batch_*, etc.).
 * @param[in]  X_parts       Local dataset partitions for this rank. Each
 *                           partition is [n_rows_i x n_features]. Individual
 *                           partitions may be empty, and a rank may have zero
 *                           local rows in aggregate (it then contributes only
 *                           to collective operations). Only the global row
 *                           count across all ranks needs to satisfy the
 *                           algorithmic precondition for the chosen init
 *                           method (e.g. `global_n >= n_clusters`).
 * @param[in]  sample_weight_parts
 *                           Optional per-partition row weights. When set, it
 *                           must contain one weight vector per data partition,
 *                           with each vector length equal to the corresponding
 *                           partition row count. When unset, all samples are
 *                           weighted equally.
 * @param[in,out] centroids  Device matrix [n_clusters x n_features]. On entry,
 *                           used as the initial centers when
 *                           `params.init == InitMethod::Array`. On return, all
 *                           RAFT comms ranks write the converged centroids;
 *                           SNMG writes them from rank 0 only.
 * @param[out] inertia       Host scalar receiving the final clustering cost on
 *                           all RAFT comms ranks, or rank 0 for SNMG.
 * @param[out] n_iter        Host scalar receiving the iteration count at which
 *                           the run terminated on all RAFT comms ranks, or
 *                           rank 0 for SNMG.
 */
template <typename DataT, typename IndexT, typename Accessor>
void mnmg_fit(
  const raft::resources& handle,
  const cuvs::cluster::kmeans::params& params,
  const std::vector<
    raft::mdspan<const DataT, raft::matrix_extent<IndexT>, raft::row_major, Accessor>>& X_parts,
  const std::optional<
    std::vector<raft::mdspan<const DataT, raft::vector_extent<IndexT>, raft::row_major, Accessor>>>&
    sample_weight_parts,
  raft::device_matrix_view<DataT, IndexT> centroids,
  raft::host_scalar_view<DataT> inertia,
  raft::host_scalar_view<IndexT> n_iter)
{
  using data_part_view_t =
    raft::mdspan<const DataT, raft::matrix_extent<IndexT>, raft::row_major, Accessor>;
  using data_batch_iterator_t =
    cuvs::spatial::knn::detail::utils::batch_load_iterator<data_part_view_t>;
  constexpr bool data_on_device = raft::is_device_mdspan_v<data_part_view_t>;

  bool use_nccl = raft::resource::is_multi_gpu(handle);
  int rank, num_ranks;
  ncclComm_t nccl_comm{};

  if (use_nccl) {
    rank      = cuvs::core::omp::get_thread_num();
    num_ranks = raft::resource::get_num_ranks(handle);
    nccl_comm = raft::resource::get_nccl_comm_for_rank(handle, rank);
  } else {
    const auto& comm = raft::resource::get_comms(handle);
    rank             = comm.get_rank();
    num_ranks        = comm.get_size();
  }

  const raft::resources& dev_res =
    use_nccl ? raft::resource::set_current_device_to_rank(handle, rank) : handle;
  mnmg_comms comms{dev_res, use_nccl, nccl_comm};

  auto stream = comms.stream();
  std::unique_ptr<rmm::cuda_stream> data_copy_stream_owner;
  rmm::cuda_stream_view data_copy_stream{stream};
  bool enable_data_prefetch = false;
  if constexpr (!data_on_device) {
    if (params.device_buffer_prefetch) {
      data_copy_stream_owner =
        std::make_unique<rmm::cuda_stream>(rmm::cuda_stream::flags::non_blocking);
      data_copy_stream     = data_copy_stream_owner->view();
      enable_data_prefetch = true;
    }
  }
  auto n_features = centroids.extent(1);
  auto n_clusters = static_cast<IndexT>(params.n_clusters);
  auto metric     = params.metric;

  RAFT_EXPECTS(metric == cuvs::distance::DistanceType::L2Expanded ||
                 metric == cuvs::distance::DistanceType::L2SqrtExpanded,
               "kmeans only supports L2Expanded or L2SqrtExpanded distance metrics.");
  RAFT_EXPECTS(n_clusters > 0, "n_clusters must be positive");
  RAFT_EXPECTS(static_cast<IndexT>(centroids.extent(0)) == n_clusters,
               "centroids.extent(0) must equal n_clusters");
  RAFT_EXPECTS(centroids.extent(1) == n_features, "centroids.extent(1) must equal n_features");
  RAFT_EXPECTS(num_ranks > 0, "num_ranks must be positive");

  IndexT n_local       = 0;
  IndexT max_part_rows = 0;
  std::vector<IndexT> part_offsets;
  part_offsets.reserve(X_parts.size() + 1);
  part_offsets.push_back(IndexT{0});
  const bool sample_weights = sample_weight_parts.has_value();
  if (sample_weights) {
    RAFT_EXPECTS(sample_weight_parts->size() == X_parts.size(),
                 "sample_weight_parts must have one entry per data partition");
  }
  for (size_t i = 0; i < X_parts.size(); ++i) {
    auto const& X_part = X_parts[i];
    RAFT_EXPECTS(static_cast<IndexT>(X_part.extent(1)) == n_features,
                 "all partitions must have the same feature count as centroids");
    auto part_rows = static_cast<IndexT>(X_part.extent(0));
    n_local += part_rows;
    max_part_rows = std::max(max_part_rows, part_rows);
    part_offsets.push_back(n_local);

    if (sample_weights) {
      RAFT_EXPECTS(static_cast<IndexT>((*sample_weight_parts)[i].extent(0)) == part_rows,
                   "each sample_weight partition must match its X partition rows");
    }
  }

  IndexT global_n{0};

  RAFT_LOG_DEBUG("MNMG KMeans fit: rank=%d/%d, n_local=%zu, n_features=%zu, n_clusters=%d",
                 rank,
                 num_ranks,
                 static_cast<size_t>(n_local),
                 static_cast<size_t>(n_features),
                 static_cast<int>(n_clusters));

  IndexT device_buffer_batch_size = static_cast<IndexT>(params.device_buffer_batch_size);
  if (device_buffer_batch_size <= 0 || device_buffer_batch_size > max_part_rows) {
    device_buffer_batch_size = std::max(max_part_rows, IndexT{1});
  }

  if (data_on_device && device_buffer_batch_size < max_part_rows) {
    RAFT_LOG_WARN(
      "MNMG KMeans: device_buffer_batch_size (%zu) ignored when partitions reside on device; using "
      "max partition size (%zu)",
      static_cast<size_t>(device_buffer_batch_size),
      static_cast<size_t>(max_part_rows));
    device_buffer_batch_size = max_part_rows;
  }

  auto rank_centroids_arr =
    use_nccl ? raft::make_device_matrix<DataT, IndexT>(dev_res, n_clusters, n_features)
             : raft::make_device_matrix<DataT, IndexT>(dev_res, 0, 0);
  auto rank_centroids = use_nccl ? rank_centroids_arr.view() : centroids;
  auto new_centroids  = raft::make_device_matrix<DataT, IndexT>(dev_res, n_clusters, n_features);
  auto centroid_sums  = raft::make_device_matrix<DataT, IndexT>(dev_res, n_clusters, n_features);
  auto weight_per_cluster    = raft::make_device_vector<DataT, IndexT>(dev_res, n_clusters);
  auto clustering_cost       = raft::make_device_vector<DataT, IndexT>(dev_res, 1);
  auto batch_clustering_cost = raft::make_device_vector<DataT, IndexT>(dev_res, 1);
  auto sqrd_norm_error_dev   = raft::make_device_scalar<DataT>(dev_res, DataT{0});
  IndexT alloc_batch_size    = device_buffer_batch_size;
  auto batch_weights         = raft::make_device_vector<DataT, IndexT>(dev_res, alloc_batch_size);
  auto minClusterAndDistance =
    raft::make_device_vector<raft::KeyValuePair<IndexT, DataT>, IndexT>(dev_res, alloc_batch_size);
  auto L2NormBatch =
    raft::make_device_vector<DataT, IndexT>(dev_res, data_on_device ? IndexT{0} : alloc_batch_size);
  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);
  rmm::device_uvector<char> workspace(0, stream);
  rmm::device_uvector<char> batch_workspace(0, stream);

  auto d_weight_scale = raft::make_device_scalar<DataT>(dev_res, DataT{1});
  if (sample_weights) {
    auto d_global_n = raft::make_device_scalar<IndexT>(dev_res, n_local);
    comms.allreduce(d_global_n.data_handle(), d_global_n.data_handle(), 1);

    auto d_wt = raft::make_device_scalar<DataT>(dev_res, DataT{0});
    for (auto const& weights : sample_weight_parts.value()) {
      auto n_weights = static_cast<IndexT>(weights.extent(0));
      if (n_weights == 0) { continue; }

      auto d_part_wt = raft::make_device_scalar<DataT>(dev_res, DataT{0});
      cuvs::cluster::kmeans::detail::weightSum(dev_res, weights, d_part_wt.view(), false);
      raft::linalg::add(d_wt.data_handle(), d_wt.data_handle(), d_part_wt.data_handle(), 1, stream);
    }

    comms.allreduce(d_wt.data_handle(), d_wt.data_handle(), 1);

    DataT h_global_wt = DataT{0};
    raft::copy(&h_global_wt, d_wt.data_handle(), 1, stream);
    raft::resource::sync_stream(dev_res);
    RAFT_EXPECTS(h_global_wt > DataT{0},
                 "invalid parameter (global sum of sample weights must be positive); "
                 "rank %d observed global sum = %g",
                 rank,
                 static_cast<double>(h_global_wt));

    const IndexT* d_global_n_ptr = d_global_n.data_handle();
    const DataT* d_wt_ptr        = d_wt.data_handle();
    raft::linalg::map(
      dev_res,
      d_weight_scale.view(),
      [d_global_n_ptr, d_wt_ptr] __device__(DataT) {
        return static_cast<DataT>(*d_global_n_ptr) / *d_wt_ptr;
      },
      raft::make_const_mdspan(d_weight_scale.view()));
  }

  auto n_init = params.n_init;
  if (params.init == cuvs::cluster::kmeans::params::InitMethod::Array && n_init != 1) {
    RAFT_LOG_DEBUG(
      "Explicit initial center position passed: performing only one init in "
      "k-means instead of n_init=%d",
      n_init);
    n_init = 1;
  }

  std::vector<IndexT> rank_counts;
  if (params.init != cuvs::cluster::kmeans::params::InitMethod::Array) {
    rank_counts = get_rank_sample_counts<IndexT>(dev_res, n_local, num_ranks, comms);
    global_n    = std::accumulate(rank_counts.begin(), rank_counts.end(), IndexT{0});
    RAFT_EXPECTS(global_n >= n_clusters,
                 "global initialization requires global row count (%zu) >= n_clusters (%zu); "
                 "rank %d has %zu local rows",
                 static_cast<size_t>(global_n),
                 static_cast<size_t>(n_clusters),
                 rank,
                 static_cast<size_t>(n_local));
  }

  auto best_centroids = n_init > 1
                          ? raft::make_device_matrix<DataT, IndexT>(dev_res, n_clusters, n_features)
                          : raft::make_device_matrix<DataT, IndexT>(dev_res, 0, 0);
  DataT best_inertia  = std::numeric_limits<DataT>::max();
  IndexT best_n_iter  = 0;

  DataT local_inertia = DataT{0};
  IndexT local_n_iter = 0;

  if (use_nccl && params.init == cuvs::cluster::kmeans::params::InitMethod::Array) {
    comms.bcast(centroids.data_handle(),
                rank_centroids.data_handle(),
                static_cast<size_t>(n_clusters) * n_features,
                CUVS_KMEANS_COMM_ROOT);
  }

  std::mt19937 gen(params.rng_state.seed);

  auto d_prior_cost = raft::make_device_scalar<DataT>(dev_res, DataT{0});
  auto d_done_flag  = raft::make_device_scalar<int>(dev_res, 0);
  auto h_done_flag  = raft::make_host_scalar<int>(0);
  auto h_norm_cache = raft::make_host_vector<DataT, IndexT>(!data_on_device ? n_local : IndexT{0});
  auto d_norms =
    raft::make_device_vector<DataT, IndexT>(dev_res, data_on_device ? n_local : IndexT{0});
  bool norms_cached = false;

  auto d_scaled_weights = raft::make_device_vector<DataT, IndexT>(
    dev_res, (data_on_device && sample_weights) ? n_local : IndexT{0});

  const DataT* d_weight_scale_ptr = d_weight_scale.data_handle();

  if constexpr (data_on_device) {
    if (sample_weights) {
      for (size_t p = 0; p < X_parts.size(); ++p) {
        auto part_rows = static_cast<IndexT>(X_parts[p].extent(0));
        if (part_rows == 0) { continue; }
        auto src = raft::make_device_vector_view<const DataT, IndexT>(
          (*sample_weight_parts)[p].data_handle(), part_rows);
        auto dst = raft::make_device_vector_view<DataT, IndexT>(
          d_scaled_weights.data_handle() + part_offsets[p], part_rows);
        raft::linalg::map(
          dev_res,
          dst,
          [d_weight_scale_ptr] __device__(DataT w) { return w * (*d_weight_scale_ptr); },
          src);
      }
    }
  }

  auto prepare_batch_weights = [&](size_t part_idx, IndexT batch_offset, IndexT cur_batch_size)
    -> raft::device_vector_view<const DataT, IndexT> {
    if (sample_weights) {
      if constexpr (data_on_device) {
        return raft::make_device_vector_view<const DataT, IndexT>(
          d_scaled_weights.data_handle() + part_offsets[part_idx] + batch_offset, cur_batch_size);
      } else {
        auto const* src = (*sample_weight_parts)[part_idx].data_handle() + batch_offset;
        raft::copy(batch_weights.data_handle(), src, cur_batch_size, stream);
        auto batch_weights_mut =
          raft::make_device_vector_view<DataT, IndexT>(batch_weights.data_handle(), cur_batch_size);
        raft::linalg::map(
          dev_res,
          batch_weights_mut,
          [d_weight_scale_ptr] __device__(DataT w) { return w * (*d_weight_scale_ptr); },
          raft::make_const_mdspan(batch_weights_mut));
      }
    }

    return raft::make_device_vector_view<const DataT, IndexT>(batch_weights.data_handle(),
                                                              cur_batch_size);
  };
  auto rank_centroids_const = raft::make_const_mdspan(rank_centroids);

  // Persist one-partition input across Lloyd iterations and inertia. Multi-partition inputs keep
  // the bounded-memory transient path.
  const bool persist_data_batches = X_parts.size() == 1 && X_parts.front().extent(0) > 0;
  std::optional<data_batch_iterator_t> persistent_data_batches;
  auto make_data_batches = [&](const data_part_view_t& X_part) -> data_batch_iterator_t {
    return persistent_data_batches
             ? *persistent_data_batches
             : data_batch_iterator_t(dev_res,
                                     X_part,
                                     static_cast<size_t>(device_buffer_batch_size),
                                     data_copy_stream,
                                     rmm::mr::get_current_device_resource_ref(),
                                     enable_data_prefetch);
  };

  for (int seed_iter = 0; seed_iter < n_init; ++seed_iter) {
    // Free the persistent batch during initialization, then retain it for Lloyd and inertia.
    persistent_data_batches.reset();

    cuvs::cluster::kmeans::params iter_params = params;
    iter_params.rng_state.seed                = gen();

    auto input_centroids_const = raft::make_device_matrix_view<const DataT, IndexT>(
      centroids.data_handle(), n_clusters, n_features);

    init_centroids_for_mg_batched<DataT, IndexT, Accessor>(dev_res,
                                                           iter_params,
                                                           device_buffer_batch_size,
                                                           X_parts,
                                                           n_features,
                                                           input_centroids_const,
                                                           rank_centroids,
                                                           workspace,
                                                           rank_counts,
                                                           global_n,
                                                           rank,
                                                           comms);

    if (persist_data_batches) {
      persistent_data_batches.emplace(dev_res,
                                      X_parts.front(),
                                      static_cast<size_t>(device_buffer_batch_size),
                                      data_copy_stream,
                                      rmm::mr::get_current_device_resource_ref(),
                                      enable_data_prefetch);
    }

    if (!sample_weights) { raft::matrix::fill(dev_res, batch_weights.view(), DataT{1}); }

    raft::matrix::fill(dev_res, d_prior_cost.view(), DataT{0});
    *h_done_flag.data_handle() = 0;

    for (local_n_iter = 1; local_n_iter <= iter_params.max_iter; ++local_n_iter) {
      if (local_n_iter > 1) {
        raft::resource::sync_stream(dev_res);
        if (*h_done_flag.data_handle()) {
          --local_n_iter;
          RAFT_LOG_DEBUG("MNMG KMeans: threshold triggered after %d iterations on rank %d",
                         static_cast<int>(local_n_iter),
                         rank);
          break;
        }
      }

      RAFT_LOG_DEBUG("MNMG KMeans: iteration %d on rank %d", local_n_iter, rank);

      raft::matrix::fill(dev_res, centroid_sums.view(), DataT{0});
      raft::matrix::fill(dev_res, weight_per_cluster.view(), DataT{0});
      raft::matrix::fill(dev_res, clustering_cost.view(), DataT{0});

      for (size_t part_idx = 0; part_idx < X_parts.size(); ++part_idx) {
        auto const& X_part = X_parts[part_idx];
        auto part_rows     = static_cast<IndexT>(X_part.extent(0));
        if (part_rows == 0) { continue; }

        auto data_batches = make_data_batches(X_part);
        auto data_it      = data_batches.begin();
        auto data_end     = data_batches.end();
        data_batches.prefetch_next_batch();

        for (; data_it != data_end; ++data_it) {
          auto const& data_batch    = *data_it;
          IndexT current_batch_size = static_cast<IndexT>(data_batch.size());
          auto batch_offset         = static_cast<IndexT>(data_batch.offset());

          auto batch_data_view = raft::make_device_matrix_view<const DataT, IndexT>(
            data_batch.data(), current_batch_size, n_features);

          auto batch_weights_view =
            prepare_batch_weights(part_idx, batch_offset, current_batch_size);

          auto norm_offset = part_offsets[part_idx] + batch_offset;
          raft::device_vector_view<const DataT, IndexT> L2NormBatch_const;
          if constexpr (data_on_device) {
            auto norm_slice = raft::make_device_vector_view<DataT, IndexT>(
              d_norms.data_handle() + norm_offset, current_batch_size);
            if (!norms_cached) {
              raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
                dev_res, batch_data_view, norm_slice);
            }
            L2NormBatch_const = raft::make_const_mdspan(norm_slice);
          } else {
            auto norm_slice = raft::make_device_vector_view<DataT, IndexT>(
              L2NormBatch.data_handle(), current_batch_size);
            if (!norms_cached) {
              raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
                dev_res, batch_data_view, norm_slice);
              raft::copy(h_norm_cache.data_handle() + norm_offset,
                         L2NormBatch.data_handle(),
                         current_batch_size,
                         stream);
            } else {
              raft::copy(L2NormBatch.data_handle(),
                         h_norm_cache.data_handle() + norm_offset,
                         current_batch_size,
                         stream);
            }
            L2NormBatch_const = raft::make_const_mdspan(norm_slice);
          }

          auto minClusterAndDistance_view =
            raft::make_device_vector_view<raft::KeyValuePair<IndexT, DataT>, IndexT>(
              minClusterAndDistance.data_handle(), current_batch_size);

          cuvs::cluster::kmeans::detail::process_batch<DataT, IndexT>(
            dev_res,
            batch_data_view,
            batch_weights_view,
            rank_centroids_const,
            metric,
            params.batch_samples,
            params.batch_centroids,
            minClusterAndDistance_view,
            L2NormBatch_const,
            L2NormBuf_OR_DistBuf,
            workspace,
            centroid_sums.view(),
            weight_per_cluster.view(),
            raft::make_device_scalar_view(clustering_cost.data_handle()),
            batch_workspace);
          data_batches.prefetch_next_batch();
        }
        if (enable_data_prefetch) { raft::resource::sync_stream(dev_res); }
      }
      norms_cached = true;

      comms.group_start();
      comms.allreduce(
        centroid_sums.data_handle(), centroid_sums.data_handle(), n_clusters * n_features);
      comms.allreduce(
        weight_per_cluster.data_handle(), weight_per_cluster.data_handle(), n_clusters);
      comms.allreduce(clustering_cost.data_handle(), clustering_cost.data_handle(), 1);
      comms.group_end();

      auto centroid_sums_const = raft::make_device_matrix_view<const DataT, IndexT>(
        centroid_sums.data_handle(), n_clusters, n_features);
      auto weight_per_cluster_const = raft::make_device_vector_view<const DataT, IndexT>(
        weight_per_cluster.data_handle(), n_clusters);

      cuvs::cluster::kmeans::detail::finalize_centroids<DataT, IndexT>(dev_res,
                                                                       centroid_sums_const,
                                                                       weight_per_cluster_const,
                                                                       rank_centroids_const,
                                                                       new_centroids.view());

      cuvs::cluster::kmeans::detail::compute_centroid_shift<DataT, IndexT>(
        dev_res,
        rank_centroids_const,
        raft::make_const_mdspan(new_centroids.view()),
        sqrd_norm_error_dev.view());

      raft::copy(
        rank_centroids.data_handle(), new_centroids.data_handle(), n_clusters * n_features, stream);

      auto d_cost_view  = raft::make_device_scalar_view<const DataT>(clustering_cost.data_handle());
      auto d_prior_view = d_prior_cost.view();
      auto d_norm_view =
        raft::make_device_scalar_view<const DataT>(sqrd_norm_error_dev.data_handle());
      auto d_done_view = d_done_flag.view();
      DataT tol        = static_cast<DataT>(params.tol);
      int iter         = static_cast<int>(local_n_iter);

      raft::linalg::map_offset(
        dev_res,
        raft::make_device_vector_view<int, int>(d_done_flag.data_handle(), 1),
        [=] __device__(int) {
          cuvs::cluster::kmeans::detail::check_convergence(
            d_cost_view, d_prior_view, d_norm_view, tol, iter, d_done_view);
          return *d_done_view.data_handle();
        });

      raft::copy(dev_res,
                 h_done_flag.view(),
                 raft::make_device_scalar_view<const int>(d_done_flag.data_handle()));
    }
    local_n_iter = std::min(local_n_iter, static_cast<IndexT>(iter_params.max_iter));

    raft::matrix::fill(dev_res, clustering_cost.view(), DataT{0});

    for (size_t part_idx = 0; part_idx < X_parts.size(); ++part_idx) {
      auto const& X_part = X_parts[part_idx];
      auto part_rows     = static_cast<IndexT>(X_part.extent(0));
      if (part_rows == 0) { continue; }

      auto data_batches = make_data_batches(X_part);
      auto data_it      = data_batches.begin();
      auto data_end     = data_batches.end();
      data_batches.prefetch_next_batch();

      for (; data_it != data_end; ++data_it) {
        auto const& data_batch    = *data_it;
        IndexT current_batch_size = static_cast<IndexT>(data_batch.size());
        auto batch_offset         = static_cast<IndexT>(data_batch.offset());

        auto batch_data_view = raft::make_device_matrix_view<const DataT, IndexT>(
          data_batch.data(), current_batch_size, n_features);

        std::optional<raft::device_vector_view<const DataT, IndexT>> batch_sw = std::nullopt;
        if (sample_weights) {
          batch_sw = prepare_batch_weights(part_idx, batch_offset, current_batch_size);
        }

        raft::matrix::fill(dev_res, batch_clustering_cost.view(), DataT{0});
        cuvs::cluster::kmeans::cluster_cost(
          dev_res,
          batch_data_view,
          rank_centroids_const,
          raft::make_device_scalar_view(batch_clustering_cost.data_handle()),
          batch_sw);

        raft::linalg::add(dev_res,
                          raft::make_const_mdspan(clustering_cost.view()),
                          raft::make_const_mdspan(batch_clustering_cost.view()),
                          clustering_cost.view());
        data_batches.prefetch_next_batch();
      }
      if (enable_data_prefetch) { raft::resource::sync_stream(dev_res); }
    }
    comms.allreduce(clustering_cost.data_handle(), clustering_cost.data_handle(), 1);
    raft::copy(&local_inertia, clustering_cost.data_handle(), 1, stream);
    raft::resource::sync_stream(dev_res);

    RAFT_LOG_DEBUG("MNMG KMeans: n_init %d/%d completed, inertia=%f, n_iter=%d on rank %d",
                   seed_iter + 1,
                   n_init,
                   static_cast<double>(local_inertia),
                   local_n_iter,
                   rank);

    if (n_init > 1 && local_inertia < best_inertia) {
      best_inertia = local_inertia;
      best_n_iter  = local_n_iter;
      raft::copy(best_centroids.data_handle(),
                 rank_centroids.data_handle(),
                 n_clusters * n_features,
                 stream);
    }
  }

  if (n_init > 1) {
    raft::copy(
      rank_centroids.data_handle(), best_centroids.data_handle(), n_clusters * n_features, stream);
    local_inertia = best_inertia;
    local_n_iter  = best_n_iter;
  }

  bool write_outputs = !use_nccl || rank == 0;
  if (write_outputs) {
    raft::copy(
      centroids.data_handle(), rank_centroids.data_handle(), n_clusters * n_features, stream);
    inertia[0] = local_inertia;
    n_iter[0]  = local_n_iter;
    raft::resource::sync_stream(dev_res);
  }
}

template <typename DataT, typename IndexT>
void batched_fit_omp(const raft::resources& clique,
                     const cuvs::cluster::kmeans::params& params,
                     raft::host_matrix_view<const DataT, IndexT> X,
                     std::optional<raft::host_vector_view<const DataT, IndexT>> sample_weight,
                     raft::device_matrix_view<DataT, IndexT> centroids,
                     raft::host_scalar_view<DataT> inertia,
                     raft::host_scalar_view<IndexT> n_iter)
{
  RAFT_EXPECTS(params.metric == cuvs::distance::DistanceType::L2Expanded ||
                 params.metric == cuvs::distance::DistanceType::L2SqrtExpanded,
               "kmeans only supports L2Expanded or L2SqrtExpanded distance metrics.");

  IndexT n_samples  = X.extent(0);
  IndexT n_features = X.extent(1);
  if (sample_weight.has_value()) {
    RAFT_EXPECTS(sample_weight->extent(0) == n_samples,
                 "invalid parameter (sample_weight extent must equal n_samples)");
  }

  raft::resource::get_nccl_comms(clique);
  int num_ranks = raft::resource::get_num_ranks(clique);

  IndexT base = n_samples / num_ranks;
  IndexT rem  = n_samples % num_ranks;

  cuvs::core::omp::check_threads(num_ranks);
  int actual_threads = 0;
#pragma omp parallel num_threads(num_ranks)
  {
#pragma omp single nowait
    {
      actual_threads = cuvs::core::omp::get_num_threads();
    }

#pragma omp barrier
    if (actual_threads == num_ranks) {
      int r          = cuvs::core::omp::get_thread_num();
      IndexT offset  = r * base + std::min<IndexT>(r, rem);
      IndexT n_local = base + (r < rem ? 1 : 0);

      std::vector<raft::host_matrix_view<const DataT, IndexT>> X_parts{
        raft::make_host_matrix_view<const DataT, IndexT>(
          X.data_handle() + offset * n_features, n_local, n_features)};

      std::optional<std::vector<raft::host_vector_view<const DataT, IndexT>>> sw_parts;
      if (sample_weight.has_value()) {
        sw_parts = std::vector<raft::host_vector_view<const DataT, IndexT>>{
          raft::make_host_vector_view<const DataT, IndexT>(sample_weight->data_handle() + offset,
                                                           n_local)};
      }

      mnmg_fit(clique, params, X_parts, sw_parts, centroids, inertia, n_iter);
    }
  }

  RAFT_EXPECTS(
    actual_threads == num_ranks,
    "OpenMP created %d threads but k-means MG requires exactly %d threads, one per rank.",
    actual_threads,
    num_ranks);
}

}  // namespace cuvs::cluster::kmeans::mg::detail

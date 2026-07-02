/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "kmeans.cuh"
#include "kmeans_common.cuh"
#include "kmeans_mg_batched_init.cuh"

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

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <raft/core/resource/comms.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace cuvs::cluster::kmeans::mg::detail {

using cuvs::core::detail::mnmg_comms;

/**
 * @brief Shared multi-GPU k-means fit core, called per rank.
 *
 * Backs both Path 1 (OMP threads sharing a clique with NCCL comms) and Path 2
 * (one rank per process with RAFT comms). The active backend is selected via
 * `raft::resource::is_multi_gpu(handle)`. Each rank streams its local shard
 * through Lloyd iterations using batched device-side reductions, allreducing
 * partial centroid sums, weights, and clustering cost at the end of each
 * iteration. Best-of-`n_init` is tracked per rank. RAFT comms ranks each write
 * their own caller-provided outputs; SNMG OMP threads share caller-provided
 * outputs, so only rank 0 writes them.
 *
 * @tparam DataT  Data / weight type (float or double)
 * @tparam IndexT Index type (int or int64_t)
 *
 * @param[in]  handle        RAFT resources. For Path 1 this is the shared
 *                           clique; for Path 2 this is the per-process handle.
 * @param[in]  params        K-means parameters (n_clusters, init, max_iter,
 *                           n_init, tol, metric, batch_*, etc.).
 * @param[in]  X_local       Host-side local shard of the dataset for this rank
 *                           [n_local x n_features]. May be empty (n_local == 0).
 * @param[in]  sample_weight Optional per-row weights for @p X_local [n_local].
 *                           When unset, all samples are weighted equally.
 * @param[in,out] centroids  Device matrix [n_clusters x n_features]. On entry,
 *                           used as the initial centers when
 *                           `params.init == InitMethod::Array`. On return,
 *                           all RAFT comms ranks write the converged centroids;
 *                           SNMG writes them from rank 0 only.
 * @param[out] inertia       Host scalar receiving the final clustering cost on
 *                           all RAFT comms ranks, or rank 0 for SNMG.
 * @param[out] n_iter        Host scalar receiving the iteration count at which
 *                           the run terminated on all RAFT comms ranks, or
 *                           rank 0 for SNMG.
 */
template <typename DataT, typename IndexT>
void mnmg_fit(const raft::resources& handle,
              const cuvs::cluster::kmeans::params& params,
              raft::host_matrix_view<const DataT, IndexT> X_local,
              std::optional<raft::host_vector_view<const DataT, IndexT>> sample_weight,
              raft::device_matrix_view<DataT, IndexT> centroids,
              raft::host_scalar_view<DataT> inertia,
              raft::host_scalar_view<IndexT> n_iter)
{
  // Setup: rank, num_ranks, dev_res, comm mechanism
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

  auto stream     = comms.stream();
  auto n_local    = X_local.extent(0);
  auto n_features = X_local.extent(1);
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
  if (sample_weight.has_value()) {
    RAFT_EXPECTS(sample_weight->extent(0) == n_local,
                 "invalid parameter (sample_weight extent must equal local row count)");
  }

  RAFT_LOG_DEBUG("MNMG KMeans fit: rank=%d/%d, n_local=%zu, n_features=%zu, n_clusters=%d",
                 rank,
                 num_ranks,
                 static_cast<size_t>(n_local),
                 static_cast<size_t>(n_features),
                 static_cast<int>(n_clusters));

  IndexT streaming_batch_size = static_cast<IndexT>(params.streaming_batch_size);
  if (streaming_batch_size <= 0 || streaming_batch_size > n_local) {
    streaming_batch_size = std::max(n_local, IndexT{1});
  }

  bool has_data = (n_local > 0);

  // Work buffers, allocated once and reused across iterations
  auto rank_centroids = raft::make_device_matrix<DataT, IndexT>(dev_res, n_clusters, n_features);
  auto new_centroids  = raft::make_device_matrix<DataT, IndexT>(dev_res, n_clusters, n_features);
  auto centroid_sums  = raft::make_device_matrix<DataT, IndexT>(dev_res, n_clusters, n_features);
  auto weight_per_cluster    = raft::make_device_vector<DataT, IndexT>(dev_res, n_clusters);
  auto clustering_cost       = raft::make_device_vector<DataT, IndexT>(dev_res, 1);
  auto batch_clustering_cost = raft::make_device_vector<DataT, IndexT>(dev_res, 1);
  auto sqrd_norm_error_dev   = raft::make_device_scalar<DataT>(dev_res, DataT{0});
  IndexT alloc_batch_size    = has_data ? streaming_batch_size : IndexT{1};
  auto batch_weights         = raft::make_device_vector<DataT, IndexT>(dev_res, alloc_batch_size);
  auto minClusterAndDistance =
    raft::make_device_vector<raft::KeyValuePair<IndexT, DataT>, IndexT>(dev_res, alloc_batch_size);
  auto L2NormBatch = raft::make_device_vector<DataT, IndexT>(dev_res, alloc_batch_size);
  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);
  rmm::device_uvector<char> workspace(0, stream);
  rmm::device_uvector<char> batch_workspace(0, stream);

  // Copy and normalize sample weights once per rank. The allreduced scale
  // terms stay on device; later batches use subviews into `scaled_weights`.
  std::optional<raft::device_scalar<IndexT>> d_global_n;
  std::optional<raft::device_scalar<DataT>> d_global_wt;
  std::optional<raft::device_vector<DataT, IndexT>> scaled_weights;
  if (sample_weight.has_value()) {
    d_global_n.emplace(raft::make_device_scalar<IndexT>(dev_res, static_cast<IndexT>(n_local)));
    d_global_wt.emplace(raft::make_device_scalar<DataT>(dev_res, DataT{0}));
    scaled_weights.emplace(raft::make_device_vector<DataT, IndexT>(dev_res, n_local));
    if (has_data) {
      raft::copy(scaled_weights->data_handle(), sample_weight->data_handle(), n_local, stream);
    }

    if (has_data) {
      cuvs::cluster::kmeans::detail::weightSum(
        dev_res, raft::make_const_mdspan(scaled_weights->view()), d_global_wt->view());
    }

    comms.group_start();
    comms.allreduce(d_global_n->data_handle(), d_global_n->data_handle(), 1);
    comms.allreduce(d_global_wt->data_handle(), d_global_wt->data_handle(), 1);
    comms.group_end();

    DataT h_global_wt = DataT{0};
    raft::copy(&h_global_wt, d_global_wt->data_handle(), 1, stream);
    raft::resource::sync_stream(dev_res);
    RAFT_EXPECTS(h_global_wt > DataT{0},
                 "invalid parameter (sum of sample weights must be positive)");

    if (has_data) {
      const IndexT* d_global_n_ptr = d_global_n->data_handle();
      const DataT* d_global_wt_ptr = d_global_wt->data_handle();
      raft::linalg::map(
        dev_res,
        scaled_weights->view(),
        [d_global_n_ptr, d_global_wt_ptr] __device__(DataT w) {
          return w * static_cast<DataT>(*d_global_n_ptr) / *d_global_wt_ptr;
        },
        raft::make_const_mdspan(scaled_weights->view()));
    }
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
  IndexT global_n = static_cast<IndexT>(n_local);
  if (params.init != cuvs::cluster::kmeans::params::InitMethod::Array) {
    rank_counts =
      get_rank_sample_counts<IndexT>(dev_res, static_cast<IndexT>(n_local), num_ranks, comms);
    global_n = std::accumulate(rank_counts.begin(), rank_counts.end(), IndexT{0});
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

  // Per-rank state (avoid races on shared host scalars in OMP)
  DataT local_inertia = DataT{0};
  IndexT local_n_iter = 0;

  std::mt19937 gen(params.rng_state.seed);

  // On-device convergence state, mirroring single-GPU `detail::fit`.
  // After centroid sums, weights, and cost are allreduced, centroid
  // finalization and shift evaluation are deterministic with identical inputs
  // on every rank, so each rank can evaluate the same convergence flag locally.
  auto d_prior_cost = raft::make_device_scalar<DataT>(dev_res, DataT{0});
  auto d_done_flag  = raft::make_device_scalar<int>(dev_res, 0);
  auto h_done_flag  = raft::make_pinned_scalar<int>(dev_res, 0);

  using data_batch_iterator_t =
    cuvs::spatial::knn::detail::utils::batch_load_iterator<decltype(X_local)>;
  std::optional<data_batch_iterator_t> data_batches_opt;
  if (has_data) {
    data_batches_opt.emplace(dev_res,
                             X_local,
                             streaming_batch_size,
                             stream,
                             rmm::mr::get_current_device_resource_ref(),
                             true);
  }

  auto h_norm_cache = raft::make_pinned_vector<DataT, IndexT>(dev_res, has_data ? n_local : 0);
  bool norms_cached = false;

  for (int seed_iter = 0; seed_iter < n_init; ++seed_iter) {
    cuvs::cluster::kmeans::params iter_params = params;
    iter_params.rng_state.seed                = gen();

    // Centroid init: selected strategy produces rank 0's centroids, then
    // broadcast to keep all ranks in lockstep.
    auto input_centroids_const = raft::make_device_matrix_view<const DataT, IndexT>(
      centroids.data_handle(), n_clusters, n_features);
    init_centroids_for_mg_batched<DataT, IndexT>(dev_res,
                                                 iter_params,
                                                 streaming_batch_size,
                                                 X_local,
                                                 input_centroids_const,
                                                 rank_centroids.view(),
                                                 workspace,
                                                 rank_counts,
                                                 global_n,
                                                 rank,
                                                 comms);
    comms.bcast(rank_centroids.data_handle(), n_clusters * n_features, 0);

    if (has_data && !sample_weight.has_value()) {
      raft::matrix::fill(dev_res, batch_weights.view(), DataT{1});
    }

    // Reset per-pass convergence state to avoid leaking it across n_init.
    raft::matrix::fill(dev_res, d_prior_cost.view(), DataT{0});
    *h_done_flag.data_handle() = 0;

    for (local_n_iter = 1; local_n_iter <= iter_params.max_iter; ++local_n_iter) {
      // Consume the previous iteration's convergence flag from pinned host.
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

      auto rank_centroids_const = raft::make_device_matrix_view<const DataT, IndexT>(
        rank_centroids.data_handle(), n_clusters, n_features);

      // Phase 1: local batch accumulation (skip if no local data)
      if (has_data) {
        auto& data_batches = *data_batches_opt;
        data_batches.reset();
        for (const auto& data_batch : data_batches) {
          IndexT current_batch_size = static_cast<IndexT>(data_batch.size());

          auto batch_data_view = raft::make_device_matrix_view<const DataT, IndexT>(
            data_batch.data(), current_batch_size, n_features);

          auto batch_offset = static_cast<IndexT>(data_batch.offset());
          auto batch_weights_view =
            sample_weight.has_value()
              ? raft::make_device_vector_view<const DataT, IndexT>(
                  scaled_weights->data_handle() + batch_offset, current_batch_size)
              : raft::make_device_vector_view<const DataT, IndexT>(batch_weights.data_handle(),
                                                                   current_batch_size);

          auto L2NormBatch_view = raft::make_device_vector_view<DataT, IndexT>(
            L2NormBatch.data_handle(), current_batch_size);

          if (!norms_cached) {
            raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
              dev_res, batch_data_view, L2NormBatch_view);
            raft::copy(h_norm_cache.data_handle() + batch_offset,
                       L2NormBatch.data_handle(),
                       current_batch_size,
                       stream);
          } else {
            raft::copy(L2NormBatch.data_handle(),
                       h_norm_cache.data_handle() + batch_offset,
                       current_batch_size,
                       stream);
          }

          auto L2NormBatch_const = raft::make_const_mdspan(L2NormBatch_view);

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
        }
        norms_cached = true;
      }

      // Phase 2: grouped allreduce
      comms.group_start();
      comms.allreduce(
        centroid_sums.data_handle(), centroid_sums.data_handle(), n_clusters * n_features);
      comms.allreduce(
        weight_per_cluster.data_handle(), weight_per_cluster.data_handle(), n_clusters);
      comms.allreduce(clustering_cost.data_handle(), clustering_cost.data_handle(), 1);
      comms.group_end();

      // Phase 3: finalize centroids
      auto centroid_sums_const = raft::make_device_matrix_view<const DataT, IndexT>(
        centroid_sums.data_handle(), n_clusters, n_features);
      auto weight_per_cluster_const = raft::make_device_vector_view<const DataT, IndexT>(
        weight_per_cluster.data_handle(), n_clusters);

      cuvs::cluster::kmeans::detail::finalize_centroids<DataT, IndexT>(dev_res,
                                                                       centroid_sums_const,
                                                                       weight_per_cluster_const,
                                                                       rank_centroids_const,
                                                                       new_centroids.view());

      // Phase 4: device-side convergence evaluation. Compute shift, run
      // `check_convergence` via `map_offset`, and shadow the flag into pinned
      // host. Consumed at top of next iteration.
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
                 raft::make_pinned_scalar_view(h_done_flag.data_handle()),
                 raft::make_device_scalar_view<const int>(d_done_flag.data_handle()));
    }
    local_n_iter = std::min(local_n_iter, static_cast<IndexT>(iter_params.max_iter));

    // Recompute inertia against the converged centroids
    raft::matrix::fill(dev_res, clustering_cost.view(), DataT{0});
    if (has_data) {
      auto rank_centroids_const = raft::make_device_matrix_view<const DataT, IndexT>(
        rank_centroids.data_handle(), n_clusters, n_features);

      auto& data_batches = *data_batches_opt;
      data_batches.reset();
      for (const auto& data_batch : data_batches) {
        IndexT current_batch_size = static_cast<IndexT>(data_batch.size());

        auto batch_data_view = raft::make_device_matrix_view<const DataT, IndexT>(
          data_batch.data(), current_batch_size, n_features);

        std::optional<raft::device_vector_view<const DataT, IndexT>> batch_sw = std::nullopt;
        if (sample_weight.has_value()) {
          batch_sw = raft::make_device_vector_view<const DataT, IndexT>(
            scaled_weights->data_handle() + static_cast<IndexT>(data_batch.offset()),
            current_batch_size);
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
      }
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

  // Final output: RAFT comms ranks are separate processes with separate output views.
  // SNMG ranks are OMP threads sharing the caller outputs, so only rank 0 writes.
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

// OpenMP wrapper for Path 1: one rank per GPU within a single process.
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
  // Verify the actual OpenMP team size before any rank enters NCCL collectives.
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

      auto X_local = raft::make_host_matrix_view<const DataT, IndexT>(
        X.data_handle() + offset * n_features, n_local, n_features);

      std::optional<raft::host_vector_view<const DataT, IndexT>> sw_local;
      if (sample_weight.has_value()) {
        sw_local = raft::make_host_vector_view<const DataT, IndexT>(
          sample_weight->data_handle() + offset, n_local);
      }

      mnmg_fit<DataT, IndexT>(clique, params, X_local, sw_local, centroids, inertia, n_iter);
    }
  }

  RAFT_EXPECTS(
    actual_threads == num_ranks,
    "OpenMP created %d threads but k-means MG requires exactly %d threads, one per rank.",
    actual_threads,
    num_ranks);
}

}  // namespace cuvs::cluster::kmeans::mg::detail

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../../core/mnmg_comms.cuh"
#include "kmeans.cuh"

#include <cuvs/cluster/kmeans.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/init.cuh>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

namespace cuvs::cluster::kmeans::mg::detail {

using cuvs::core::detail::mnmg_comms;

template <typename IndexT>
IndexT get_global_kmeanspp_init_sample_size(const cuvs::cluster::kmeans::params& params,
                                            IndexT global_n,
                                            IndexT n_clusters)
{
  IndexT default_init_size = std::min(static_cast<IndexT>(std::int64_t{3} * n_clusters), global_n);
  IndexT init_sample_size  = params.init_size > 0
                               ? std::min(static_cast<IndexT>(params.init_size), global_n)
                               : default_init_size;
  return std::max(init_sample_size, n_clusters);
}

template <typename IndexT>
std::vector<IndexT> sample_unique_global_indices(IndexT n_samples,
                                                 IndexT sample_size,
                                                 std::uint64_t seed)
{
  std::mt19937 gen(seed);
  std::vector<IndexT> sampled;
  sampled.reserve(static_cast<std::size_t>(sample_size));
  std::unordered_set<IndexT> selected;
  selected.reserve(static_cast<std::size_t>(sample_size));

  for (IndexT j = n_samples - sample_size; j < n_samples; ++j) {
    std::uniform_int_distribution<IndexT> dis(IndexT{0}, j);
    IndexT candidate = dis(gen);
    if (selected.find(candidate) != selected.end()) { candidate = j; }
    selected.insert(candidate);
    sampled.push_back(candidate);
  }

  return sampled;
}

template <typename IndexT>
std::vector<IndexT> get_rank_sample_counts(raft::resources const& handle,
                                           IndexT n_local,
                                           int num_ranks,
                                           const mnmg_comms& comms)
{
  auto stream    = comms.stream();
  auto d_n_local = raft::make_device_scalar<IndexT>(handle, n_local);
  auto d_counts  = raft::make_device_vector<IndexT, IndexT>(handle, static_cast<IndexT>(num_ranks));

  comms.allgather(d_n_local.data_handle(), d_counts.data_handle(), 1);

  std::vector<IndexT> h_counts(static_cast<std::size_t>(num_ranks));
  raft::copy(h_counts.data(), d_counts.data_handle(), static_cast<size_t>(num_ranks), stream);
  raft::resource::sync_stream(handle);
  return h_counts;
}

template <typename IndexT>
std::vector<IndexT> get_rank_offsets(const std::vector<IndexT>& rank_counts)
{
  std::vector<IndexT> offsets(rank_counts.size() + 1, IndexT{0});
  std::partial_sum(rank_counts.begin(), rank_counts.end(), offsets.begin() + 1);
  return offsets;
}

template <typename IndexT>
struct owned_sample_index {
  IndexT sample_idx;
  IndexT local_idx;
};

template <typename IndexT>
std::vector<owned_sample_index<IndexT>> get_owned_sample_indices(
  const std::vector<IndexT>& sample_ids, const std::vector<IndexT>& rank_counts, int rank)
{
  auto rank_offsets  = get_rank_offsets(rank_counts);
  IndexT local_begin = rank_offsets[static_cast<std::size_t>(rank)];
  IndexT local_end   = rank_offsets[static_cast<std::size_t>(rank + 1)];

  std::vector<owned_sample_index<IndexT>> owned_samples;
  owned_samples.reserve(sample_ids.size());
  for (IndexT sample_idx = 0; sample_idx < static_cast<IndexT>(sample_ids.size()); ++sample_idx) {
    IndexT global_idx = sample_ids[static_cast<std::size_t>(sample_idx)];
    if (global_idx >= local_begin && global_idx < local_end) {
      owned_samples.push_back({sample_idx, global_idx - local_begin});
    }
  }

  return owned_samples;
}

template <typename IndexT>
std::vector<IndexT> broadcast_sampled_global_indices(raft::resources const& handle,
                                                     std::uint64_t seed,
                                                     IndexT global_n,
                                                     IndexT sample_size,
                                                     int rank,
                                                     int root,
                                                     const mnmg_comms& comms)
{
  RAFT_EXPECTS(sample_size > 0, "global initialization sample size must be positive");
  RAFT_EXPECTS(sample_size <= global_n,
               "global initialization sample size (%zu) must be <= global row count (%zu)",
               static_cast<size_t>(sample_size),
               static_cast<size_t>(global_n));

  auto d_sample_ids = raft::make_device_vector<IndexT, IndexT>(handle, sample_size);
  std::vector<IndexT> h_sample_ids(static_cast<std::size_t>(sample_size));
  if (rank == root) {
    h_sample_ids = sample_unique_global_indices(global_n, sample_size, seed);
    raft::copy(
      handle,
      raft::make_device_vector_view<IndexT, IndexT>(d_sample_ids.data_handle(), sample_size),
      raft::make_host_vector_view<const IndexT, IndexT>(h_sample_ids.data(), sample_size));
  }

  comms.bcast(d_sample_ids.data_handle(), static_cast<size_t>(sample_size), root);

  raft::copy(
    handle,
    raft::make_host_vector_view<IndexT, IndexT>(h_sample_ids.data(), sample_size),
    raft::make_device_vector_view<const IndexT, IndexT>(d_sample_ids.data_handle(), sample_size));
  raft::resource::sync_stream(handle);
  return h_sample_ids;
}

template <typename DataT, typename IndexT>
raft::device_matrix<DataT, IndexT> sample_global_host_rows(
  raft::resources const& handle,
  const cuvs::cluster::kmeans::params& params,
  raft::host_matrix_view<const DataT, IndexT> X_local,
  IndexT sample_size,
  int rank,
  int root,
  const std::vector<IndexT>& rank_counts,
  IndexT global_n,
  const mnmg_comms& comms)
{
  auto n_features = X_local.extent(1);
  auto sample_ids = broadcast_sampled_global_indices(
    handle, params.rng_state.seed, global_n, sample_size, rank, root, comms);
  auto owned_samples = get_owned_sample_indices(sample_ids, rank_counts, rank);

  auto sampled_rows = raft::make_device_matrix<DataT, IndexT>(handle, sample_size, n_features);
  raft::matrix::fill(handle, sampled_rows.view(), DataT{0});

  for (auto const& owned_sample : owned_samples) {
    raft::copy(handle,
               raft::make_device_vector_view<DataT, IndexT>(
                 sampled_rows.data_handle() + owned_sample.sample_idx * n_features, n_features),
               raft::make_host_vector_view<const DataT, IndexT>(
                 X_local.data_handle() + owned_sample.local_idx * n_features, n_features));
  }

  comms.allreduce(sampled_rows.data_handle(), sampled_rows.data_handle(), sampled_rows.size());
  return sampled_rows;
}

template <typename DataT, typename IndexT>
void init_centroids_for_mg_batched(raft::resources const& handle,
                                   const cuvs::cluster::kmeans::params& params,
                                   IndexT /*streaming_batch_size*/,
                                   raft::host_matrix_view<const DataT, IndexT> X_local,
                                   raft::device_matrix_view<const DataT, IndexT> initial_centroids,
                                   raft::device_matrix_view<DataT, IndexT> centroids,
                                   rmm::device_uvector<char>& workspace,
                                   const std::vector<IndexT>& rank_counts,
                                   IndexT global_n,
                                   int rank,
                                   const mnmg_comms& comms)
{
  constexpr int root = 0;
  auto stream        = comms.stream();
  auto n_features    = X_local.extent(1);
  auto n_clusters    = static_cast<IndexT>(params.n_clusters);

  if (params.init == cuvs::cluster::kmeans::params::InitMethod::Array) {
    if (rank == root) {
      raft::copy(centroids.data_handle(),
                 initial_centroids.data_handle(),
                 static_cast<size_t>(n_clusters) * n_features,
                 stream);
    }
    return;
  }

  if (params.init == cuvs::cluster::kmeans::params::InitMethod::Random) {
    auto sampled_rows = sample_global_host_rows<DataT, IndexT>(
      handle, params, X_local, n_clusters, rank, root, rank_counts, global_n, comms);
    raft::copy(centroids.data_handle(), sampled_rows.data_handle(), sampled_rows.size(), stream);
  } else if (params.init == cuvs::cluster::kmeans::params::InitMethod::KMeansPlusPlus) {
    IndexT init_sample_size = get_global_kmeanspp_init_sample_size(params, global_n, n_clusters);
    auto init_sample        = sample_global_host_rows<DataT, IndexT>(
      handle, params, X_local, init_sample_size, rank, root, rank_counts, global_n, comms);

    if (rank == root) {
      // KMeans++ seeding runs on root, so root must have enough device memory
      // for the full initialization sample plus the seeding workspace.
      auto init_view = raft::make_const_mdspan(init_sample.view());
      if (params.oversampling_factor == 0) {
        cuvs::cluster::kmeans::detail::kmeansPlusPlus<DataT, IndexT>(
          handle, params, init_view, centroids, workspace);
      } else {
        cuvs::cluster::kmeans::detail::initScalableKMeansPlusPlus<DataT, IndexT>(
          handle, params, init_view, centroids, workspace);
      }
    }
  } else {
    THROW("unknown initialization method to select initial centers");
  }
}

}  // namespace cuvs::cluster::kmeans::mg::detail

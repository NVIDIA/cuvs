/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "kmeans_common.cuh"

#include <cuda/std/tuple>
#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/linalg/coalesced_reduction.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/matrix/select_k.cuh>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

namespace cuvs::cluster::kmeans::detail {

/** Dense row-major IVF lists for approximate Lloyd assignments. No distance kernels are added:
 * routing, grouping, gathering, list-local 1NN and winner reduction use existing primitives.
 * Rebuild after each centroid update; all ranks index the complete replicated centroid matrix.
 */
template <typename DataT, typename IndexT>
class dense_ivf_assignment {
 public:
  using kvp         = raft::KeyValuePair<IndexT, DataT>;
  using matrix_view = raft::device_matrix_view<const DataT, IndexT>;

  dense_ivf_assignment(raft::resources const& handle, params const& p) : handle_(handle), params_(p)
  {
    RAFT_EXPECTS(p.metric == cuvs::distance::DistanceType::L2Expanded,
                 "Dense IVF assignment supports L2Expanded only");
    RAFT_EXPECTS(p.ivf_n_lists > 0 && p.ivf_n_probes > 0 && p.ivf_n_probes <= p.ivf_n_lists,
                 "Dense IVF requires 0 < ivf_n_probes <= ivf_n_lists");
    RAFT_EXPECTS(p.ivf_max_iter > 0 && p.ivf_batch_samples > 0,
                 "Dense IVF iteration and query batch limits must be positive");
  }

  void build(matrix_view centroids)
  {
    const auto stream = raft::resource::get_cuda_stream(handle_);
    auto policy       = raft::resource::get_thrust_policy(handle_);
    n_centroids_      = centroids.extent(0);
    dim_              = centroids.extent(1);
    RAFT_EXPECTS(n_centroids_ > 0 && dim_ > 0, "Dense IVF requires nonempty centroids");
    n_lists_ = std::min<IndexT>(params_.ivf_n_lists, n_centroids_);
    coarse_.resize(static_cast<std::size_t>(n_lists_) * dim_, stream);

    // A separate single-device handle prevents recursive multi-GPU dispatch when this runs
    // inside a rank. Coarse fitting always uses exhaustive assignments, never another IVF.
    raft::resources local_handle;
    raft::resource::set_cuda_stream(local_handle, stream);
    params coarse_params;
    coarse_params.n_clusters = static_cast<int>(n_lists_);
    coarse_params.init       = params::Random;
    coarse_params.max_iter   = params_.ivf_max_iter;
    coarse_params.tol        = std::numeric_limits<double>::min();
    coarse_params.rng_state  = params_.rng_state;
    DataT inertia{};
    IndexT iterations{};
    cuvs::cluster::kmeans::fit(local_handle,
                               coarse_params,
                               centroids,
                               std::nullopt,
                               raft::make_device_matrix_view(coarse_.data(), n_lists_, dim_),
                               raft::make_host_scalar_view(&inertia),
                               raft::make_host_scalar_view(&iterations));

    auto assigned = nearest(centroids, coarse_view());
    list_ids_.resize(n_centroids_, stream);
    centroid_ids_.resize(n_centroids_, stream);
    assigned.visit_indices(
      [&](auto ids) { thrust::copy(policy, ids, ids + n_centroids_, list_ids_.data()); });
    thrust::sequence(policy, centroid_ids_.begin(), centroid_ids_.end());
    // Stable order preserves original centroid-ID tie breaking inside each list.
    thrust::stable_sort_by_key(policy, list_ids_.begin(), list_ids_.end(), centroid_ids_.begin());
    list_offsets_.resize(n_lists_ + 1, stream);
    offsets(list_ids_.data(), n_centroids_, list_offsets_.data(), host_list_offsets_);
    dense_centroids_.resize(static_cast<std::size_t>(n_centroids_) * dim_, stream);
    raft::matrix::gather(centroids.data_handle(),
                         dim_,
                         n_centroids_,
                         centroid_ids_.data(),
                         n_centroids_,
                         dense_centroids_.data(),
                         stream.get());
    IndexT nonempty = 0;
    for (IndexT i = 0; i < n_lists_; ++i) {
      nonempty += host_list_offsets_[i + 1] != host_list_offsets_[i];
    }
    n_probes_ = std::min<IndexT>(params_.ivf_n_probes, nonempty);
  }

  MinClusterAndDistanceResult<DataT, IndexT> search(matrix_view queries)
  {
    RAFT_EXPECTS(n_centroids_ > 0, "Build the dense IVF lists before searching");
    RAFT_EXPECTS(queries.extent(1) == dim_, "Dense IVF query dimension mismatch");
    const auto stream = raft::resource::get_cuda_stream(handle_);
    auto policy       = raft::resource::get_thrust_policy(handle_);
    const IndexT rows = queries.extent(0);
    output_.resize(rows, stream);
    const IndexT batch =
      std::min<IndexT>(params_.ivf_batch_samples, std::numeric_limits<IndexT>::max() / n_probes_);
    for (IndexT begin = 0; begin < rows; begin += std::min(batch, rows - begin)) {
      const IndexT count = std::min(batch, rows - begin);
      const IndexT pairs = count * n_probes_;
      auto x             = raft::make_device_matrix_view<const DataT, IndexT>(
        queries.data_handle() + static_cast<std::size_t>(begin) * dim_, count, dim_);
      routing_distances_.resize(static_cast<std::size_t>(count) * n_lists_, stream);
      auto route = raft::make_device_matrix_view(routing_distances_.data(), count, n_lists_);
      pairwise_distance_kmeans<DataT, IndexT>(
        handle_, x, coarse_view(), route, cuvs::distance::DistanceType::L2Expanded);
      auto* list_offsets = list_offsets_.data();
      auto lists         = n_lists_;
      // Empty lists cannot consume the entire probe budget (e.g. duplicate coarse centers).
      auto first = thrust::make_counting_iterator<std::size_t>(0);
      thrust::transform(policy,
                        routing_distances_.begin(),
                        routing_distances_.end(),
                        first,
                        routing_distances_.begin(),
                        [=] __device__(DataT distance, std::size_t i) {
                          auto list = i % lists;
                          return list_offsets[list] == list_offsets[list + 1]
                                   ? std::numeric_limits<DataT>::infinity()
                                   : distance;
                        });
      probe_ids_.resize(pairs, stream);
      probe_distances_.resize(pairs, stream);
      raft::matrix::select_k<DataT, IndexT>(
        handle_,
        raft::make_device_matrix_view<const DataT, int64_t>(
          routing_distances_.data(), count, n_lists_),
        std::nullopt,
        raft::make_device_matrix_view<DataT, int64_t>(probe_distances_.data(), count, n_probes_),
        raft::make_device_matrix_view<IndexT, int64_t>(probe_ids_.data(), count, n_probes_),
        true);

      pair_order_.resize(pairs, stream);
      thrust::sequence(policy, pair_order_.begin(), pair_order_.end());
      thrust::stable_sort_by_key(policy, probe_ids_.begin(), probe_ids_.end(), pair_order_.begin());
      query_offsets_.resize(n_lists_ + 1, stream);
      offsets(probe_ids_.data(), pairs, query_offsets_.data(), host_query_offsets_);
      auto probes    = n_probes_;
      auto query_ids = thrust::make_transform_iterator(
        pair_order_.data(), [=] __host__ __device__(IndexT i) { return i / probes; });
      gathered_queries_.resize(static_cast<std::size_t>(pairs) * dim_, stream);
      raft::matrix::gather(
        x.data_handle(), dim_, count, query_ids, pairs, gathered_queries_.data(), stream.get());
      pair_results_.resize(pairs, stream);
      for (IndexT list = 0; list < n_lists_; ++list) {
        auto q_start = host_query_offsets_[list];
        auto q_count = host_query_offsets_[list + 1] - q_start;
        if (q_count == 0) { continue; }
        auto c_start = host_list_offsets_[list];
        auto c_count = host_list_offsets_[list + 1] - c_start;
        RAFT_EXPECTS(c_count > 0, "Dense IVF selected an empty list");
        auto grouped = raft::make_device_matrix_view<const DataT, IndexT>(
          gathered_queries_.data() + static_cast<std::size_t>(q_start) * dim_, q_count, dim_);
        auto candidates = raft::make_device_matrix_view<const DataT, IndexT>(
          dense_centroids_.data() + static_cast<std::size_t>(c_start) * dim_, c_count, dim_);
        // This is the same top_1_nn dispatch (including cuTile) used by standard KMeans.
        auto result        = nearest(grouped, candidates);
        auto* original_ids = centroid_ids_.data() + c_start;
        auto dest =
          thrust::make_permutation_iterator(pair_results_.data(), pair_order_.data() + q_start);
        result.visit_indices([&](auto ids) {
          result.visit_distances([&](auto distances) {
            auto values = thrust::make_zip_iterator(cuda::std::make_tuple(ids, distances));
            thrust::transform(policy, values, values + q_count, dest, remap_result{original_ids});
          });
        });
      }
      raft::linalg::coalescedReduction(
        output_.data() + begin,
        pair_results_.data(),
        n_probes_,
        count,
        kvp{std::numeric_limits<IndexT>::max(), std::numeric_limits<DataT>::infinity()},
        stream.get(),
        false,
        raft::identity_op{},
        raft::argmin_op{},
        raft::identity_op{});
    }
    cuvs::distance::detail::Top1nnPlan<IndexT> plan{};
    plan.m                = rows;
    plan.output_layout    = cuvs::distance::detail::Top1nnOutputLayout::KeyValuePair;
    plan.output_bytes     = static_cast<std::size_t>(rows) * sizeof(kvp);
    plan.output_alignment = alignof(kvp);
    return MinClusterAndDistanceResult<DataT, IndexT>::from_output(plan, rows, output_.data());
  }

  // Must be accessible to CUDA-generated Thrust launch stubs.
  struct remap_result {
    IndexT const* original_ids;
    template <typename Tuple>
    __device__ kvp operator()(Tuple value) const
    {
      return {original_ids[cuda::std::get<0>(value)], cuda::std::get<1>(value)};
    }
  };

 private:
  matrix_view coarse_view() const
  {
    return raft::make_device_matrix_view<const DataT, IndexT>(coarse_.data(), n_lists_, dim_);
  }

  MinClusterAndDistanceResult<DataT, IndexT> nearest(matrix_view x, matrix_view y)
  {
    const auto stream = raft::resource::get_cuda_stream(handle_);
    norms_.resize(x.extent(0), stream);
    auto norm = raft::make_device_vector_view(norms_.data(), x.extent(0));
    raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(handle_, x, norm);
    return minClusterAndDistanceCompute<DataT, IndexT>(handle_,
                                                       x,
                                                       y,
                                                       native_output_,
                                                       raft::make_const_mdspan(norm),
                                                       norm_or_distance_,
                                                       cuvs::distance::DistanceType::L2Expanded,
                                                       params_.batch_samples,
                                                       params_.batch_centroids,
                                                       workspace_);
  }

  void offsets(IndexT const* keys, IndexT count, IndexT* output, std::vector<IndexT>& host)
  {
    auto policy = raft::resource::get_thrust_policy(handle_);
    auto first  = thrust::make_counting_iterator<IndexT>(0);
    thrust::lower_bound(policy, keys, keys + count, first, first + n_lists_ + 1, output);
    host.resize(n_lists_ + 1);
    raft::update_host(host.data(), output, host.size(), raft::resource::get_cuda_stream(handle_));
    raft::resource::sync_stream(handle_);
  }

  raft::resources const& handle_;
  params params_;
  IndexT n_centroids_{}, dim_{}, n_lists_{}, n_probes_{};
  std::vector<IndexT> host_list_offsets_, host_query_offsets_;
#define IVF_BUFFER(Type, Name) \
  rmm::device_uvector<Type> Name { 0, raft::resource::get_cuda_stream(handle_) }
  IVF_BUFFER(DataT, coarse_);
  IVF_BUFFER(DataT, dense_centroids_);
  IVF_BUFFER(DataT, gathered_queries_);
  IVF_BUFFER(DataT, routing_distances_);
  IVF_BUFFER(DataT, probe_distances_);
  IVF_BUFFER(DataT, norms_);
  IVF_BUFFER(DataT, norm_or_distance_);
  IVF_BUFFER(IndexT, list_ids_);
  IVF_BUFFER(IndexT, centroid_ids_);
  IVF_BUFFER(IndexT, list_offsets_);
  IVF_BUFFER(IndexT, probe_ids_);
  IVF_BUFFER(IndexT, pair_order_);
  IVF_BUFFER(IndexT, query_offsets_);
  IVF_BUFFER(kvp, pair_results_);
  IVF_BUFFER(kvp, output_);
  IVF_BUFFER(char, native_output_);
  IVF_BUFFER(char, workspace_);
#undef IVF_BUFFER
};

}  // namespace cuvs::cluster::kmeans::detail

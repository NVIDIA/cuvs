/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../../core/gather_launchers.hpp"
#include "../../../core/omp_wrapper.hpp"

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/matrix/gather.cuh>

namespace cuvs::neighbors::experimental::scann::detail {

template <typename T, typename IdxT>
struct gather_functor {
  auto operator()(raft::resources const& res,
                  raft::device_matrix_view<const T, int64_t> dataset,
                  raft::device_vector_view<const IdxT, int64_t> cluster_ids,
                  raft::device_matrix_view<T, int64_t> cluster_vecs,
                  cudaStream_t stream)
  {
    cuvs::detail::gather(
      res, raft::make_const_mdspan(dataset), raft::make_const_mdspan(cluster_ids), cluster_vecs);
  }

  auto operator()(raft::resources const& res,
                  raft::host_matrix_view<const T, int64_t> dataset,
                  raft::device_vector_view<const IdxT, int64_t> cluster_ids,
                  raft::device_matrix_view<T, int64_t> cluster_vecs,
                  cudaStream_t stream)
  {
    auto h_cluster_ids = raft::make_host_vector<IdxT, int64_t>(cluster_ids.extent(0));

    raft::copy(res, h_cluster_ids.view(), raft::make_const_mdspan(cluster_ids));

    auto pinned_cluster =
      raft::make_host_matrix<T, int64_t>(cluster_vecs.extent(0), cluster_vecs.extent(1));

    raft::resource::sync_stream(res, stream);

    int n_threads = std::min<int>(cuvs::core::omp::get_max_threads(), 32);

#pragma omp parallel for num_threads(n_threads)
    for (int i = 0; i < h_cluster_ids.extent(0); i++) {
      // std::
      memcpy(pinned_cluster.data_handle() + i * pinned_cluster.extent(1),
             dataset.data_handle() + h_cluster_ids(i) * dataset.extent(1),
             sizeof(T) * dataset.extent(1));
    }

    raft::copy(
      res,
      raft::make_device_vector_view(cluster_vecs.data_handle(), pinned_cluster.size()),
      raft::make_host_vector_view<const T>(pinned_cluster.data_handle(), pinned_cluster.size()));
    raft::resource::sync_stream(res, stream);
  }
};

}  // namespace cuvs::neighbors::experimental::scann::detail

/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs_internal/neighbors/naive_knn.cuh>
#include <raft/core/resource/cuda_stream.hpp>

#include <cuvs/distance/distance.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng.cuh>

#include <cuda/stream>
#include <rmm/device_uvector.hpp>

namespace cuvs::neighbors {

template <typename IdxT>
struct RefineInputs {
  IdxT n_queries;
  IdxT n_rows;
  IdxT dim;
  IdxT k;   // after refinement
  IdxT k0;  // initial k before refinement (k0 >= k).
  cuvs::distance::DistanceType metric;
  bool host_data;
};

/** Helper class to allocate arrays and generate input data for refinement test and benchmark. */
template <typename DataT, typename DistanceT, typename IdxT>
class RefineHelper {
 public:
  RefineHelper(const raft::resources& handle, RefineInputs<IdxT> params)
    : handle_(handle),
      stream_(resource::get_cuda_stream(handle)),
      p(params),
      dataset(handle),
      queries(handle),
      refined_distances(handle),
      refined_indices(handle),
      candidates(handle),
      dataset_host(handle),
      queries_host(handle),
      candidates_host(handle),
      refined_distances_host(handle),
      refined_indices_host(handle)
  {
    raft::random::RngState rng(1234ULL);

    dataset = raft::make_device_matrix<DataT, IdxT>(handle_, p.n_rows, p.dim);
    queries = raft::make_device_matrix<DataT, IdxT>(handle_, p.n_queries, p.dim);
    if constexpr (std::is_same<DataT, float>{}) {
      raft::random::uniform(
        handle, rng, dataset.data_handle(), dataset.size(), DataT(-10.0), DataT(10.0));
      raft::random::uniform(
        handle, rng, queries.data_handle(), queries.size(), DataT(-10.0), DataT(10.0));
    } else {
      raft::random::uniformInt(
        handle, rng, dataset.data_handle(), dataset.size(), DataT(1), DataT(20));
      raft::random::uniformInt(
        handle, rng, queries.data_handle(), queries.size(), DataT(1), DataT(20));
    }

    refined_distances = raft::make_device_matrix<DistanceT, IdxT>(handle_, p.n_queries, p.k);
    refined_indices   = raft::make_device_matrix<IdxT, IdxT>(handle_, p.n_queries, p.k);

    // Generate candidate vectors
    {
      candidates = raft::make_device_matrix<IdxT, IdxT>(handle_, p.n_queries, p.k0);
      rmm::device_uvector<DistanceT> distances_tmp(p.n_queries * p.k0, stream_);
      naive_knn<DistanceT, DataT, IdxT>(handle_,
                                        distances_tmp.data(),
                                        candidates.data_handle(),
                                        queries.data_handle(),
                                        dataset.data_handle(),
                                        p.n_queries,
                                        p.n_rows,
                                        p.dim,
                                        p.k0,
                                        p.metric);
      resource::sync_stream(handle_, stream_);
    }

    if (p.host_data) {
      dataset_host    = raft::make_host_matrix<DataT, IdxT>(p.n_rows, p.dim);
      queries_host    = raft::make_host_matrix<DataT, IdxT>(p.n_queries, p.dim);
      candidates_host = raft::make_host_matrix<IdxT, IdxT>(p.n_queries, p.k0);

      raft::copy(dataset_host.data_handle(), dataset.data_handle(), dataset.size(), stream_);
      raft::copy(queries_host.data_handle(), queries.data_handle(), queries.size(), stream_);
      raft::copy(
        candidates_host.data_handle(), candidates.data_handle(), candidates.size(), stream_);

      refined_distances_host = raft::make_host_matrix<DistanceT, IdxT>(p.n_queries, p.k);
      refined_indices_host   = raft::make_host_matrix<IdxT, IdxT>(p.n_queries, p.k);
      resource::sync_stream(handle_, stream_);
    }

    // Generate ground thruth for testing.
    {
      rmm::device_uvector<DistanceT> distances_dev(p.n_queries * p.k, stream_);
      rmm::device_uvector<IdxT> indices_dev(p.n_queries * p.k, stream_);
      naive_knn<DistanceT, DataT, IdxT>(handle_,
                                        distances_dev.data(),
                                        indices_dev.data(),
                                        queries.data_handle(),
                                        dataset.data_handle(),
                                        p.n_queries,
                                        p.n_rows,
                                        p.dim,
                                        p.k,
                                        p.metric);
      true_refined_distances_host.resize(p.n_queries * p.k);
      true_refined_indices_host.resize(p.n_queries * p.k);
      raft::copy(true_refined_indices_host.data(), indices_dev.data(), indices_dev.size(), stream_);
      raft::copy(
        true_refined_distances_host.data(), distances_dev.data(), distances_dev.size(), stream_);
      resource::sync_stream(handle_, stream_);
    }
  }

 public:
  RefineInputs<IdxT> p;
  const raft::resources& handle_;
  cuda::stream_ref stream_;

  raft::device_matrix<DataT, IdxT, row_major> dataset;
  raft::device_matrix<DataT, IdxT, row_major> queries;
  raft::device_matrix<IdxT, IdxT, row_major> candidates;  // Neighbor candidate indices
  raft::device_matrix<IdxT, IdxT, row_major> refined_indices;
  raft::device_matrix<DistanceT, IdxT, row_major> refined_distances;

  raft::host_matrix<DataT, IdxT, row_major> dataset_host;
  raft::host_matrix<DataT, IdxT, row_major> queries_host;
  raft::host_matrix<IdxT, IdxT, row_major> candidates_host;
  raft::host_matrix<IdxT, IdxT, row_major> refined_indices_host;
  raft::host_matrix<DistanceT, IdxT, row_major> refined_distances_host;

  std::vector<IdxT> true_refined_indices_host;
  std::vector<DistanceT> true_refined_distances_host;
};
}  // namespace cuvs::neighbors

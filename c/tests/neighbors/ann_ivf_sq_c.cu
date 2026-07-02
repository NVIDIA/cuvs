/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda.h>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>

#include "neighbors/ann_utils.cuh"
#include <cuvs/neighbors/ivf_sq.h>

extern "C" void run_ivf_sq(cuvsResources_t res,
                           int64_t n_rows,
                           int64_t n_queries,
                           int64_t n_dim,
                           uint32_t n_neighbors,
                           float* index_data,
                           float* query_data,
                           float* distances_data,
                           int64_t* neighbors_data,
                           cuvsDistanceType metric,
                           size_t n_probes,
                           size_t n_lists);

template <typename T>
void generate_random_data(raft::handle_t const& handle, T* devPtr, size_t size)
{
  raft::random::RngState r(1234ULL);
  raft::random::uniform(handle, r, devPtr, size, T(0.1), T(2.0));
};

template <typename T, typename IdxT>
void recall_eval(raft::handle_t const& handle,
                 T* query_data,
                 T* index_data,
                 IdxT* neighbors,
                 T* distances,
                 size_t n_queries,
                 size_t n_rows,
                 size_t n_dim,
                 size_t n_neighbors,
                 cuvsDistanceType metric,
                 size_t n_probes,
                 size_t n_lists)
{
  auto distances_ref = raft::make_device_matrix<T, IdxT>(handle, n_queries, n_neighbors);
  auto neighbors_ref = raft::make_device_matrix<IdxT, IdxT>(handle, n_queries, n_neighbors);
  cuvs::neighbors::naive_knn<T, T, IdxT>(
    handle,
    distances_ref.data_handle(),
    neighbors_ref.data_handle(),
    query_data,
    index_data,
    n_queries,
    n_rows,
    n_dim,
    n_neighbors,
    static_cast<cuvs::distance::DistanceType>((uint16_t)metric));

  size_t size = n_queries * n_neighbors;
  std::vector<IdxT> neighbors_h(size);
  std::vector<T> distances_h(size);
  std::vector<IdxT> neighbors_ref_h(size);
  std::vector<T> distances_ref_h(size);

  auto stream = raft::resource::get_cuda_stream(handle);
  raft::copy(neighbors_h.data(), neighbors, size, stream);
  raft::copy(distances_h.data(), distances, size, stream);
  raft::copy(neighbors_ref_h.data(), neighbors_ref.data_handle(), size, stream);
  raft::copy(distances_ref_h.data(), distances_ref.data_handle(), size, stream);
  raft::resource::sync_stream(handle);

  double min_recall = static_cast<double>(n_probes) / static_cast<double>(n_lists);
  ASSERT_TRUE(cuvs::neighbors::eval_neighbours(neighbors_ref_h,
                                               neighbors_h,
                                               distances_ref_h,
                                               distances_h,
                                               n_queries,
                                               n_neighbors,
                                               0.001,
                                               min_recall));
};

TEST(IvfSqC, BuildSearch)
{
  int64_t n_rows       = 8096;
  int64_t n_queries    = 128;
  int64_t n_dim        = 32;
  uint32_t n_neighbors = 8;

  raft::handle_t handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  cuvsDistanceType metric = L2Expanded;
  size_t n_probes         = 20;
  size_t n_lists          = 1024;

  rmm::device_uvector<float> index_data(n_rows * n_dim, stream);
  rmm::device_uvector<float> query_data(n_queries * n_dim, stream);
  rmm::device_uvector<int64_t> neighbors_data(n_queries * n_neighbors, stream);
  rmm::device_uvector<float> distances_data(n_queries * n_neighbors, stream);

  generate_random_data(handle, index_data.data(), n_rows * n_dim);
  generate_random_data(handle, query_data.data(), n_queries * n_dim);

  cuvsResources_t res;
  cuvsResourcesCreate(&res);
  cuvsStreamSet(res, stream);

  run_ivf_sq(res,
             n_rows,
             n_queries,
             n_dim,
             n_neighbors,
             index_data.data(),
             query_data.data(),
             distances_data.data(),
             neighbors_data.data(),
             metric,
             n_probes,
             n_lists);

  recall_eval(handle,
              query_data.data(),
              index_data.data(),
              neighbors_data.data(),
              distances_data.data(),
              n_queries,
              n_rows,
              n_dim,
              n_neighbors,
              metric,
              n_probes,
              n_lists);

  cuvsResourcesDestroy(res);
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/bitmap.hpp>
#include <cuvs/neighbors/brute_force.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/make_blobs.cuh>

#include <iostream>

void load_dataset(const raft::device_resources& res, float* data_ptr, int n_vectors, int dim)
{
  raft::random::RngState rng(1234ULL);
  raft::random::uniform(res, rng, data_ptr, n_vectors * dim, 0.1f, 2.0f);
}

int main()
{
  using namespace cuvs::neighbors;
  using dataset_dtype  = float;
  using indexing_dtype = int64_t;
  auto dim             = 128;
  auto n_vectors       = 90;
  auto n_queries       = 100;
  auto k               = 5;

  // ... build index ...
  raft::device_resources res;
  brute_force::index_params index_params;
  brute_force::search_params search_params;
  auto dataset = raft::make_device_matrix<dataset_dtype, indexing_dtype>(res, n_vectors, dim);
  auto queries = raft::make_device_matrix<dataset_dtype, indexing_dtype>(res, n_queries, dim);

  load_dataset(res, dataset.data_handle(), n_vectors, dim);
  load_dataset(res, queries.data_handle(), n_queries, dim);
  auto index = brute_force::build(res, index_params, raft::make_const_mdspan(dataset.view()));

  // Load a list of all the samples that will get filtered
  std::vector<indexing_dtype> removed_indices_host = {2, 13, 21, 8};
  auto removed_indices_device =
    raft::make_device_vector<indexing_dtype, indexing_dtype>(res, removed_indices_host.size());
  // Copy this list to device
  raft::copy(removed_indices_device.data_handle(),
             removed_indices_host.data(),
             removed_indices_host.size(),
             raft::resource::get_cuda_stream(res));

  // Create a bitmap with the list of samples to filter.
  cuvs::core::bitset<uint32_t, indexing_dtype> removed_indices_bitset(
    res, removed_indices_device.view(), n_queries * n_vectors);
  cuvs::core::bitmap_view<const uint32_t, indexing_dtype> removed_indices_bitmap(
    removed_indices_bitset.data(), n_queries, n_vectors);

  // Use a `bitmap_filter` in the `brute_force::search` function call.
  auto bitmap_filter = cuvs::neighbors::filtering::bitmap_filter(removed_indices_bitmap);

  auto neighbors = raft::make_device_matrix<indexing_dtype, indexing_dtype>(res, n_queries, k);
  auto distances = raft::make_device_matrix<dataset_dtype, indexing_dtype>(res, n_queries, k);
  std::cout << "Searching..." << std::endl;
  brute_force::search(res,
                      search_params,
                      index,
                      raft::make_const_mdspan(queries.view()),
                      neighbors.view(),
                      distances.view(),
                      bitmap_filter);
  std::cout << "Success!" << std::endl;
  return 0;
}

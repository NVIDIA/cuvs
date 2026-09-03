/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/random/sample_without_replacement.cuh>
#include <raft/util/cudart_utils.hpp>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>

#include <fstream>

// Fill dataset and queries with synthetic data.
void generate_dataset(raft::device_resources const& dev_resources,
                      raft::device_matrix_view<float, int64_t> dataset,
                      raft::device_matrix_view<float, int64_t> queries)
{
  auto labels = raft::make_device_vector<int64_t, int64_t>(dev_resources, dataset.extent(0));
  raft::random::make_blobs(dev_resources, dataset, labels.view());
  raft::random::RngState r(1234ULL);
  raft::random::uniform(dev_resources,
                        r,
                        raft::make_device_vector_view(queries.data_handle(), queries.size()),
                        -1.0f,
                        1.0f);
}

// Copy the results to host and print a few samples
template <typename IdxT>
void print_results(raft::device_resources const& dev_resources,
                   raft::device_matrix_view<IdxT, int64_t> neighbors,
                   raft::device_matrix_view<float, int64_t> distances)
{
  int64_t topk        = neighbors.extent(1);
  auto neighbors_host = raft::make_host_matrix<IdxT, int64_t>(neighbors.extent(0), topk);
  auto distances_host = raft::make_host_matrix<float, int64_t>(distances.extent(0), topk);

  cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources).get();

  raft::copy(neighbors_host.data_handle(), neighbors.data_handle(), neighbors.size(), stream);
  raft::copy(distances_host.data_handle(), distances.data_handle(), distances.size(), stream);

  // The calls to RAFT algorithms and  raft::copy is asynchronous.
  // We need to sync the stream before accessing the data.
  raft::resource::sync_stream(dev_resources, stream);

  for (int query_id = 0; query_id < 2; query_id++) {
    std::cout << "Query " << query_id << " neighbor indices: ";
    raft::print_host_vector("", &neighbors_host(query_id, 0), topk, std::cout);
    std::cout << "Query " << query_id << " neighbor distances: ";
    raft::print_host_vector("", &distances_host(query_id, 0), topk, std::cout);
  }
}

/** Subsample the dataset to create a training set*/
raft::device_matrix<float, int64_t> subsample(
  raft::device_resources const& dev_resources,
  raft::device_matrix_view<const float, int64_t> dataset,
  raft::device_vector_view<const int64_t, int64_t> data_indices,
  float fraction)
{
  int64_t n_samples = dataset.extent(0);
  int64_t n_dim     = dataset.extent(1);
  int64_t n_train   = n_samples * fraction;
  auto trainset     = raft::make_device_matrix<float, int64_t>(dev_resources, n_train, n_dim);

  int seed = 137;
  raft::random::RngState rng(seed);
  auto train_indices = raft::make_device_vector<int64_t>(dev_resources, n_train);

  raft::random::sample_without_replacement(
    dev_resources, rng, data_indices, std::nullopt, train_indices.view(), std::nullopt);

  raft::matrix::copy_rows(
    dev_resources, dataset, trainset.view(), raft::make_const_mdspan(train_indices.view()));

  return trainset;
}

template <typename T, typename idxT>
raft::device_matrix<T, idxT> read_bin_dataset(raft::device_resources const& dev_resources,
                                              std::string fname,
                                              int max_N = INT_MAX)
{
  // Read datafile in
  std::ifstream datafile(fname, std::ifstream::binary);
  uint32_t N;
  uint32_t dim;
  datafile.read((char*)&N, sizeof(uint32_t));
  datafile.read((char*)&dim, sizeof(uint32_t));

  if (N > max_N) N = max_N;
  printf("Read in file - N:%u, dim:%u\n", N, dim);
  std::vector<T> data;
  data.resize((size_t)N * (size_t)dim);
  datafile.read(reinterpret_cast<char*>(data.data()), (size_t)N * (size_t)dim * sizeof(T));
  datafile.close();

  auto dataset = raft::make_device_matrix<T, idxT>(dev_resources, N, dim);
  raft::copy(dataset.data_handle(),
             data.data(),
             data.size(),
             raft::resource::get_cuda_stream(dev_resources));

  return dataset;
}

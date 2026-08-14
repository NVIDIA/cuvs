/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <cstdlib>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/make_blobs.cuh>

#include <cuvs/neighbors/scann.hpp>

#include <rmm/mr/pool_memory_resource.hpp>

#include "common.cuh"

template <typename T>
void scann_build_and_write(raft::device_resources const& dev_resources,
                           raft::device_matrix_view<const T, int64_t> dataset)
{
  using namespace cuvs::neighbors::experimental;

  // use default index parameters
  scann::index_params index_params;

  index_params.n_leaves            = 1000;
  index_params.kmeans_n_rows_train = 10000;
  index_params.partitioning_eta    = 2;
  index_params.soar_lambda         = 1.5;

  index_params.pq_dim          = 2;
  index_params.pq_bits         = 4;
  index_params.pq_n_rows_train = 10000;

  std::cout << "Building ScaNN index" << std::endl;

  auto start = std::chrono::system_clock::now();
  auto index = scann::build(dev_resources, index_params, dataset);
  auto end   = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;

  // TODO - output statistics about the index

  std::cout << "Time to build index: " << elapsed_seconds.count() << "s\n";

  // Output index to files in /tmp directory
  serialize(dev_resources, "/tmp", index);
}

int main(int argc, char* argv[])
{
  raft::device_resources dev_resources;

  // Set pool memory resource with 1 GiB initial pool size. All allocations use
  // the same pool.
  rmm::mr::pool_memory_resource pool_mr(rmm::mr::get_current_device_resource_ref(),
                                        1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(pool_mr);

  // Create input arrays.
  int64_t n_samples = 10000;
  int64_t n_dim     = 64;
  int64_t n_queries = 10;

  auto dataset = raft::make_device_matrix<float, int64_t>(dev_resources, n_samples, n_dim);
  auto queries = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, n_dim);

  generate_dataset(dev_resources, dataset.view(), queries.view());

  // Simple build example to create ScaNN index and write to file
  scann_build_and_write<float>(dev_resources, raft::make_const_mdspan(dataset.view()));
}

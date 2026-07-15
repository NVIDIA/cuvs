/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <filesystem>
#include <memory>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <string>

#include <cuvs/neighbors/hnsw.hpp>

#include <rmm/mr/pool_memory_resource.hpp>

#include "common.cuh"

void hnsw_build_search_ace(raft::device_resources const& dev_resources,
                           raft::device_matrix_view<const float, int64_t> dataset,
                           raft::device_matrix_view<const float, int64_t> queries)
{
  using namespace cuvs::neighbors;

  int64_t topk      = 12;
  int64_t n_queries = queries.extent(0);

  // HNSW ACE build requires the dataset to be on the host
  auto dataset_host = raft::make_host_matrix<float, int64_t>(dataset.extent(0), dataset.extent(1));
  raft::copy(dataset_host.data_handle(),
             dataset.data_handle(),
             dataset.extent(0) * dataset.extent(1),
             raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);

  // HNSW index parameters
  hnsw::index_params hnsw_params;
  hnsw_params.metric    = cuvs::distance::DistanceType::L2Expanded;
  hnsw_params.hierarchy = hnsw::HnswHierarchy::GPU;

  // Parameters for GPU accelerated HNSW index building
  auto ace_params = hnsw::graph_build_params::ace_params();
  // Set the number of partitions. Small values might improve recall but potentially degrade
  // performance and increase memory usage. Partitions should not be too small to prevent issues in
  // KNN graph construction. The partition size is on average 2 * (n_rows / npartitions) * dim *
  // sizeof(T). 2 is because of the core and augmented vectors. Please account for imbalance in the
  // partition sizes (up to 3x in our tests).
  ace_params.npartitions = 4;
  // Set the directory to store the ACE build artifacts. This should be the fastest disk in the
  // system and hold enough space for twice the dataset, final graph, and label mapping.
  ace_params.build_dir = "/tmp/hnsw_ace_build";
  // Set whether to use disk-based storage for ACE build. When true, enables disk-based operations
  // for memory-efficient graph construction. If not set, the index will be built in memory if the
  // graph fits in host and GPU memory, and on disk otherwise.
  ace_params.use_disk            = true;
  hnsw_params.graph_build_params = ace_params;
  // Set M parameter to control the graph degree (graph_degree = m * 2, intermediate_graph_degree =
  // m * 3). Higher values work for higher intrinsic dimensionality and/or high recall, low values
  // can work for datasets with low intrinsic dimensionality and/or low recalls. Higher values lead
  // to higher memory consumption.
  hnsw_params.M = 32;
  // Set the index quality for the ACE build. Bigger values increase the index quality. At some
  // point, increasing this will no longer improve the quality.
  hnsw_params.ef_construction = 120;

  // Build the HNSW index using ACE
  std::cout << "Building HNSW index using ACE" << std::endl;
  auto hnsw_index =
    hnsw::build(dev_resources, hnsw_params, raft::make_const_mdspan(dataset_host.view()));

  // For disk-based indexes, the build function serializes the index to disk
  // We need to deserialize it before searching
  std::string hnsw_index_path = hnsw_index->file_path();
  std::cout << "Deserializing HNSW index from disk for search" << std::endl;
  hnsw::index<float>* hnsw_index_deserialized = nullptr;
  hnsw::deserialize(dev_resources,
                    hnsw_params,
                    hnsw_index_path,
                    dataset.extent(1),
                    hnsw_params.metric,
                    &hnsw_index_deserialized);

  // Prepare queries on host for HNSW search
  auto queries_host = raft::make_host_matrix<float, int64_t>(n_queries, queries.extent(1));
  raft::copy(queries_host.data_handle(),
             queries.data_handle(),
             n_queries * queries.extent(1),
             raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);

  // Create output arrays for HNSW search (uses uint64_t indices)
  auto indices_host   = raft::make_host_matrix<uint64_t, int64_t>(n_queries, topk);
  auto distances_host = raft::make_host_matrix<float, int64_t>(n_queries, topk);

  // Configure search parameters
  hnsw::search_params search_params;
  search_params.ef          = std::max(200, static_cast<int>(topk) * 2);
  search_params.num_threads = 1;

  // Search the HNSW index
  std::cout << "Searching HNSW index" << std::endl;
  hnsw::search(dev_resources,
               search_params,
               *hnsw_index_deserialized,
               queries_host.view(),
               indices_host.view(),
               distances_host.view());

  // Convert results to device for printing
  auto neighbors = raft::make_device_matrix<uint32_t>(dev_resources, n_queries, topk);
  auto distances = raft::make_device_matrix<float>(dev_resources, n_queries, topk);

  // Convert HNSW uint64_t indices to uint32_t
  auto neighbors_host = raft::make_host_matrix<uint32_t, int64_t>(n_queries, topk);
  for (int64_t i = 0; i < n_queries; i++) {
    for (int64_t j = 0; j < topk; j++) {
      neighbors_host(i, j) = static_cast<uint32_t>(indices_host(i, j));
    }
  }

  // Copy results to device
  raft::copy(neighbors.data_handle(),
             neighbors_host.data_handle(),
             n_queries * topk,
             raft::resource::get_cuda_stream(dev_resources));
  raft::copy(distances.data_handle(),
             distances_host.data_handle(),
             n_queries * topk,
             raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);

  print_results(dev_resources, neighbors.view(), distances.view());
}

int main()
{
  raft::device_resources dev_resources;

  // Set pool memory resource with 1 GiB initial pool size. All allocations use the same pool.
  rmm::mr::pool_memory_resource pool_mr(rmm::mr::get_current_device_resource_ref(),
                                        1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(pool_mr);

  // Alternatively, one could define a pool allocator for temporary arrays (used within RAFT
  // algorithms). In that case only the internal arrays would use the pool, any other allocation
  // uses the default RMM memory resource. Here is how to change the workspace memory resource to
  // a pool with 2 GiB upper limit.
  // raft::resource::set_workspace_to_pool_resource(dev_resources, 2 * 1024 * 1024 * 1024ull);

  // Create input arrays.
  int64_t n_samples = 10000;
  int64_t n_dim     = 90;
  int64_t n_queries = 10;
  auto dataset      = raft::make_device_matrix<float, int64_t>(dev_resources, n_samples, n_dim);
  auto queries      = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, n_dim);
  generate_dataset(dev_resources, dataset.view(), queries.view());

  // HNSW ACE build and search example.
  hnsw_build_search_ace(dev_resources,
                        raft::make_const_mdspan(dataset.view()),
                        raft::make_const_mdspan(queries.view()));
}

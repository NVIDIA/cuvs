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

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/hnsw.hpp>

#include <rmm/mr/pool_memory_resource.hpp>

#include "common.cuh"

void cagra_build_search_ace(raft::device_resources const& dev_resources,
                            raft::device_matrix_view<const float, int64_t> dataset,
                            raft::device_matrix_view<const float, int64_t> queries)
{
  using namespace cuvs::neighbors;

  int64_t topk      = 12;
  int64_t n_queries = queries.extent(0);

  // create output arrays
  auto neighbors = raft::make_device_matrix<uint32_t>(dev_resources, n_queries, topk);
  auto distances = raft::make_device_matrix<float>(dev_resources, n_queries, topk);

  // CAGRA index parameters
  cagra::index_params index_params;
  index_params.intermediate_graph_degree = 128;
  index_params.graph_degree              = 64;

  // ACE index parameters
  auto ace_params = cagra::graph_build_params::ace_params();
  // Set the number of partitions. Small values might improve recall but potentially degrade
  // performance and increase memory usage. Partitions should not be too small to prevent issues in
  // KNN graph construction. The partition size is on average 2 * (n_rows / npartitions) * dim *
  // sizeof(T). 2 is because of the core and augmented vectors. Please account for imbalance in the
  // partition sizes (up to 3x in our tests).
  ace_params.npartitions = 4;
  // Set the index quality for the ACE build. Bigger values increase the index quality. At some
  // point, increasing this will no longer improve the quality.
  ace_params.ef_construction = 120;
  // Set the directory to store the ACE build artifacts. This should be the fastest disk in the
  // system and hold enough space for twice the dataset, final graph, and label mapping.
  ace_params.build_dir = "/tmp/ace_build";
  // Set whether to use disk-based storage for ACE build. When true, enables disk-based operations
  // for memory-efficient graph construction. If not set, the index will be built in memory if the
  // graph fits in host and GPU memory, and on disk otherwise.
  ace_params.use_disk             = true;
  index_params.graph_build_params = ace_params;

  // ACE requires the dataset to be on the host
  auto dataset_host = raft::make_host_matrix<float, int64_t>(dataset.extent(0), dataset.extent(1));
  raft::copy(dataset_host.data_handle(),
             dataset.data_handle(),
             dataset.extent(0) * dataset.extent(1),
             raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);
  auto dataset_host_view = raft::make_host_matrix_view<const float, int64_t, raft::row_major>(
    dataset_host.data_handle(), dataset_host.extent(0), dataset_host.extent(1));

  std::cout << "Building CAGRA index (search graph)" << std::endl;
  auto index = cagra::build(dev_resources, index_params, dataset_host_view);
  // In-memory build of ACE provides the index in memory, so we can search it directly using
  // cagra::search

  // On-disk build of ACE stores the reordered dataset, the dataset mapping, and the graph on disk.
  // The index is not directly usable for CAGRA search. Convert to HNSW for search operations.

  // Convert CAGRA index to HNSW
  // For disk-based indices: serializes CAGRA to HNSW format on disk, returns an index with file
  // descriptor For in-memory indices: creates HNSW index in memory
  std::cout << "Converting CAGRA index to HNSW" << std::endl;
  hnsw::index_params hnsw_params;
  hnsw_params.hierarchy = hnsw::HnswHierarchy::GPU;  // Offload hierarchy construction to GPU
  auto hnsw_index       = hnsw::from_cagra(dev_resources, hnsw_params, index);

  // HNSW search requires host matrices
  auto queries_host = raft::make_host_matrix<float, int64_t>(n_queries, queries.extent(1));
  raft::copy(queries_host.data_handle(),
             queries.data_handle(),
             n_queries * queries.extent(1),
             raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);

  // HNSW search outputs uint64_t indices
  auto indices_hnsw_host   = raft::make_host_matrix<uint64_t, int64_t>(n_queries, topk);
  auto distances_hnsw_host = raft::make_host_matrix<float, int64_t>(n_queries, topk);

  hnsw::search_params hnsw_search_params;
  hnsw_search_params.ef          = std::max(200, static_cast<int>(topk) * 2);
  hnsw_search_params.num_threads = 1;

  // If the HNSW index is in memory, search directly
  // std::cout << "HNSW index in memory. Searching..." << std::endl;
  // hnsw::search(dev_resources,
  //              hnsw_search_params,
  //              *hnsw_index,
  //              queries_host.view(),
  //              indices_hnsw_host.view(),
  //              distances_hnsw_host.view());

  // If the HNSW index is stored on disk, deserialize it for searching
  std::cout << "HNSW index is stored on disk." << std::endl;

  // For disk-based indices, the HNSW index file path can be obtained via file_path()
  std::string hnsw_index_path = hnsw_index->file_path();
  std::cout << "HNSW index file location: " << hnsw_index_path << std::endl;
  std::cout << "Deserializing HNSW index from disk for search." << std::endl;

  hnsw::index<float>* hnsw_index_raw = nullptr;
  hnsw::deserialize(
    dev_resources, hnsw_params, hnsw_index_path, index.dim(), index.metric(), &hnsw_index_raw);

  std::unique_ptr<hnsw::index<float>> hnsw_index_deserialized(hnsw_index_raw);

  std::cout << "Searching HNSW index." << std::endl;
  hnsw::search(dev_resources,
               hnsw_search_params,
               *hnsw_index_deserialized,
               queries_host.view(),
               indices_hnsw_host.view(),
               distances_hnsw_host.view());

  // Convert HNSW uint64_t indices back to uint32_t for printing
  auto neighbors_host = raft::make_host_matrix<uint32_t, int64_t>(n_queries, topk);
  for (int64_t i = 0; i < n_queries; i++) {
    for (int64_t j = 0; j < topk; j++) {
      neighbors_host(i, j) = static_cast<uint32_t>(indices_hnsw_host(i, j));
    }
  }

  // Copy results to device
  raft::copy(neighbors.data_handle(),
             neighbors_host.data_handle(),
             n_queries * topk,
             raft::resource::get_cuda_stream(dev_resources));
  raft::copy(distances.data_handle(),
             distances_hnsw_host.data_handle(),
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

  // ACE build and search example.
  cagra_build_search_ace(dev_resources,
                         raft::make_const_mdspan(dataset.view()),
                         raft::make_const_mdspan(queries.view()));
}

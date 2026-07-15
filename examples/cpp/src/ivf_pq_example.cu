/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"

#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/refine.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>

#include <rmm/mr/pool_memory_resource.hpp>

#include <cstdint>

void ivf_pq_build_search(raft::device_resources const& dev_resources,
                         raft::device_matrix_view<const float, int64_t> dataset,
                         raft::device_matrix_view<const float, int64_t> queries)
{
  using namespace cuvs::neighbors;  // NOLINT

  ivf_pq::index_params index_params;
  index_params.n_lists                  = 1024;
  index_params.kmeans_trainset_fraction = 0.1;
  index_params.metric                   = cuvs::distance::DistanceType::L2Expanded;
  index_params.pq_bits                  = 8;
  index_params.pq_dim                   = 2;

  std::cout << "Building IVF-PQ index" << std::endl;
  auto index = ivf_pq::build(dev_resources, index_params, dataset);

  std::cout << "Number of clusters " << index.n_lists() << ", number of vectors added to index "
            << index.size() << std::endl;

  // Set search parameters.
  ivf_pq::search_params search_params;
  search_params.n_probes = 50;
  // Set the internal search precision to 16-bit floats;
  // usually, this improves the performance at a slight cost to the recall.
  search_params.internal_distance_dtype = CUDA_R_16F;
  search_params.lut_dtype               = CUDA_R_16F;

  // Create output arrays.
  int64_t topk      = 10;
  int64_t n_queries = queries.extent(0);
  auto neighbors    = raft::make_device_matrix<int64_t>(dev_resources, n_queries, topk);
  auto distances    = raft::make_device_matrix<float>(dev_resources, n_queries, topk);

  // Search K nearest neighbors for each of the queries.
  ivf_pq::search(dev_resources, search_params, index, queries, neighbors.view(), distances.view());

  // Re-ranking operation: refine the initial search results by computing exact distances
  int64_t topk_refined = 7;
  auto neighbors_refined =
    raft::make_device_matrix<int64_t>(dev_resources, n_queries, topk_refined);
  auto distances_refined = raft::make_device_matrix<float>(dev_resources, n_queries, topk_refined);

  // Note, refinement requires the original dataset and the queries.
  // Don't forget to specify the same distance metric as used by the index.
  cuvs::neighbors::refine(dev_resources,
                          dataset,
                          queries,
                          raft::make_const_mdspan(neighbors.view()),
                          neighbors_refined.view(),
                          distances_refined.view(),
                          index.metric());

  // Show both the original and the refined results
  std::cout << std::endl << "Original results:" << std::endl;
  print_results(dev_resources, neighbors.view(), distances.view());
  std::cout << std::endl << "Refined results:" << std::endl;
  print_results(dev_resources, neighbors_refined.view(), distances_refined.view());
}

int main()
{
  raft::device_resources dev_resources;

  // Set pool memory resource with 1 GiB initial pool size. All allocations use the same pool.
  rmm::mr::pool_memory_resource pool_mr(rmm::mr::get_current_device_resource_ref(),
                                        1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(pool_mr);

  // Create input arrays.
  int64_t n_samples = 10000;
  int64_t n_dim     = 3;
  int64_t n_queries = 10;
  auto dataset      = raft::make_device_matrix<float, int64_t>(dev_resources, n_samples, n_dim);
  auto queries      = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, n_dim);
  generate_dataset(dev_resources, dataset.view(), queries.view());

  // Simple build and search example.
  ivf_pq_build_search(dev_resources,
                      raft::make_const_mdspan(dataset.view()),
                      raft::make_const_mdspan(queries.view()));
}

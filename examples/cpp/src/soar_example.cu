/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/cluster/soar.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/mr/pool_memory_resource.hpp>

#include <cstdint>
#include <iostream>

/** Number of rows whose secondary cluster differs from their primary one. */
int64_t count_spilled(raft::device_resources const& dev_resources,
                      raft::device_vector_view<const uint32_t, int64_t> labels,
                      raft::device_vector_view<const uint32_t, int64_t> soar_labels)
{
  auto h_labels      = raft::make_host_vector<uint32_t, int64_t>(labels.extent(0));
  auto h_soar_labels = raft::make_host_vector<uint32_t, int64_t>(soar_labels.extent(0));
  auto stream        = raft::resource::get_cuda_stream(dev_resources);

  raft::copy(h_labels.data_handle(), labels.data_handle(), labels.size(), stream);
  raft::copy(h_soar_labels.data_handle(), soar_labels.data_handle(), soar_labels.size(), stream);
  raft::resource::sync_stream(dev_resources, stream);

  int64_t n_spilled = 0;
  for (int64_t i = 0; i < labels.extent(0); ++i) {
    if (h_soar_labels(i) != h_labels(i)) { ++n_spilled; }
  }
  return n_spilled;
}

void soar_predict_example(raft::device_resources const& dev_resources,
                          raft::device_matrix_view<const float, int64_t> dataset,
                          raft::device_matrix_view<const float, int64_t> centroids,
                          raft::device_vector_view<const uint32_t, int64_t> labels)
{
  // Default lambda = 1. Larger values penalize secondary centroids whose residual is aligned with
  // the primary residual, favoring more complementary assignments.
  cuvs::cluster::soar::params params;

  auto soar_labels = raft::make_device_vector<uint32_t, int64_t>(dev_resources, dataset.extent(0));

  cuvs::cluster::soar::predict(
    dev_resources, params, dataset, centroids, labels, soar_labels.view());

  // A row keeps its primary label when no other centroid is worth spilling to.
  auto n_spilled =
    count_spilled(dev_resources, labels, raft::make_const_mdspan(soar_labels.view()));

  std::cout << "Spilled " << n_spilled << " of " << dataset.extent(0)
            << " rows to a secondary cluster" << std::endl;
}

int main()
{
  raft::device_resources dev_resources;

  // Set pool memory resource with 1 GiB initial pool size. All allocations use the same pool.
  rmm::mr::pool_memory_resource pool_mr(rmm::mr::get_current_device_resource_ref(),
                                        1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(pool_mr);

  int64_t n_samples  = 10000;
  int64_t n_dim      = 64;
  int64_t n_clusters = 100;

  // Far fewer blobs than k-means clusters: the 100 learned partitions subdivide the 10 dense
  // regions, creating internal partition boundaries. blob_labels is required by make_blobs but
  // unused.
  int64_t n_blobs  = 10;
  auto dataset     = raft::make_device_matrix<float, int64_t>(dev_resources, n_samples, n_dim);
  auto blob_labels = raft::make_device_vector<int64_t, int64_t>(dev_resources, n_samples);
  raft::random::make_blobs(dev_resources, dataset.view(), blob_labels.view(), n_blobs);

  auto dataset_view = raft::make_const_mdspan(dataset.view());

  // SOAR needs centroids and a primary label per row, so k-means runs first.
  cuvs::cluster::kmeans::balanced_params kmeans_params;
  auto centroids = raft::make_device_matrix<float, int64_t>(dev_resources, n_clusters, n_dim);
  auto labels    = raft::make_device_vector<uint32_t, int64_t>(dev_resources, n_samples);

  cuvs::cluster::kmeans::fit(dev_resources, kmeans_params, dataset_view, centroids.view());
  cuvs::cluster::kmeans::predict(dev_resources,
                                 kmeans_params,
                                 dataset_view,
                                 raft::make_const_mdspan(centroids.view()),
                                 labels.view());

  soar_predict_example(dev_resources,
                       dataset_view,
                       raft::make_const_mdspan(centroids.view()),
                       raft::make_const_mdspan(labels.view()));
}

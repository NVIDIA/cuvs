/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/cluster/kmeans.hpp>

#include "../../src/cluster/detail/kmeans_common.cuh"

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/memory_stats_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace cuvs::cluster::kmeans {

struct BatchConfig {
  char const* name;
  int batch_samples;
  int batch_centroids;
};

constexpr int test_n_samples  = 8;
constexpr int test_n_clusters = 8;
constexpr int test_n_features = 2;

template <typename IndexT>
std::size_t run_predict_with_batching(raft::resources const& handle, BatchConfig config)
{
  SCOPED_TRACE(config.name);

  using DataT = float;

  constexpr IndexT n_samples  = test_n_samples;
  constexpr IndexT n_clusters = test_n_clusters;
  constexpr IndexT n_features = test_n_features;

  // The nearest centroids span all three centroid tiles. The batched configurations also have
  // partial final tiles, exercising the tile-local index adjustment and the minima merge.
  constexpr std::array<DataT, n_samples * n_features> h_x{0.0f,
                                                          0.0f,
                                                          0.5f,
                                                          0.5f,
                                                          3.0f,
                                                          3.0f,
                                                          3.25f,
                                                          3.25f,
                                                          6.0f,
                                                          6.0f,
                                                          5.75f,
                                                          5.75f,
                                                          2.0f,
                                                          2.0f,
                                                          4.0f,
                                                          4.0f};
  constexpr std::array<DataT, n_clusters * n_features> h_centroids{0.0f,
                                                                   0.0f,
                                                                   100.0f,
                                                                   100.0f,
                                                                   200.0f,
                                                                   200.0f,
                                                                   3.0f,
                                                                   3.0f,
                                                                   400.0f,
                                                                   400.0f,
                                                                   500.0f,
                                                                   500.0f,
                                                                   6.0f,
                                                                   6.0f,
                                                                   700.0f,
                                                                   700.0f};
  constexpr std::array<IndexT, n_samples> expected_clusters{0, 0, 3, 3, 6, 6, 3, 3};
  constexpr DataT expected_inertia = 4.75f;

  auto stream    = raft::resource::get_cuda_stream(handle);
  auto x         = raft::make_device_matrix<DataT, IndexT>(handle, n_samples, n_features);
  auto centroids = raft::make_device_matrix<DataT, IndexT>(handle, n_clusters, n_features);
  auto labels    = raft::make_device_vector<IndexT, IndexT>(handle, n_samples);

  raft::update_device(x.data_handle(), h_x.data(), h_x.size(), stream);
  raft::update_device(centroids.data_handle(), h_centroids.data(), h_centroids.size(), stream);

  params kmeans_params;
  kmeans_params.n_clusters      = n_clusters;
  kmeans_params.batch_samples   = config.batch_samples;
  kmeans_params.batch_centroids = config.batch_centroids;

  DataT inertia                  = 0;
  std::size_t total_device_bytes = 0;
  {
    raft::memory_stats_resources tracked_handle{handle};
    predict(tracked_handle,
            kmeans_params,
            raft::make_const_mdspan(x.view()),
            std::optional<raft::device_vector_view<const DataT, IndexT>>{std::nullopt},
            raft::make_const_mdspan(centroids.view()),
            labels.view(),
            false,
            raft::make_host_scalar_view(&inertia));
    total_device_bytes = tracked_handle.get_bytes_total_allocated().device_global;
  }

  std::array<IndexT, n_samples> h_labels;
  raft::update_host(h_labels.data(), labels.data_handle(), h_labels.size(), stream);
  raft::resource::sync_stream(handle);

  EXPECT_NEAR(inertia, expected_inertia, 1e-4f);
  for (std::size_t i = 0; i < h_labels.size(); ++i) {
    EXPECT_EQ(h_labels[i], expected_clusters[i]) << "sample " << i;
  }

  return total_device_bytes;
}

TEST(KMeansPredict, BatchParametersPreserveResultsAndReduceUnfusedAllocations)
{
  raft::resources handle;
  constexpr std::array<BatchConfig, 4> batch_configs{{
    {"no batching", 0, 0},
    {"samples only", 3, 0},
    {"centroids only", 0, 3},
    {"samples and centroids", 3, 3},
  }};

  auto unbatched_int_bytes   = run_predict_with_batching<int>(handle, batch_configs.front());
  auto unbatched_int64_bytes = run_predict_with_batching<int64_t>(handle, batch_configs.front());

  // predict selects fused or unfused 1-NN according to the architecture heuristic. The batching
  // parameters only affect the unfused path, so every GPU checks the results while allocation
  // reductions are required only when this problem shape dispatches to unfused 1-NN.
  const auto fused_path =
    detail::use_fused<float, int, int>(handle,
                                       test_n_samples,
                                       test_n_clusters,
                                       test_n_features,
                                       cuvs::distance::DistanceType::L2Expanded);
  const bool uses_unfused_path = !detail::uses_fused_distance_nn(fused_path);

  for (std::size_t i = 1; i < batch_configs.size(); ++i) {
    auto config      = batch_configs[i];
    auto int_bytes   = run_predict_with_batching<int>(handle, config);
    auto int64_bytes = run_predict_with_batching<int64_t>(handle, config);

    if (uses_unfused_path) {
      // Verify that batching uses less memory than the unbatched path.
      EXPECT_LT(int_bytes, unbatched_int_bytes) << config.name;
      EXPECT_LT(int64_bytes, unbatched_int64_bytes) << config.name;
    }
  }
}

}  // namespace cuvs::cluster::kmeans

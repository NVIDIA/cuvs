/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Benchmark: brute force vs CAGRA-based cluster assignment for IVF training.
 * Compares time to assign N vectors to K clusters (nearest centroid) using
 * (1) brute force 1-NN and (2) CAGRA build on centroids + k=1 search.
 */
#include <benchmark/benchmark.h>

// kmeans_balanced.cuh is under cpp/src/; CUVS_CLUSTER_ASSIGNMENT_BENCH adds that to include path
#include <cluster/kmeans_balanced.cuh>
#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/init.cuh>
#include <raft/random/rng.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuvs/neighbors/cagra.hpp>
#include <rmm/device_uvector.hpp>

#include <optional>

namespace {

using namespace cuvs::cluster::kmeans_balanced;

void init_random_data(raft::resources const& handle,
                      float* X,
                      int64_t n_rows,
                      int64_t dim,
                      float* centroids,
                      int64_t n_clusters)
{
  raft::random::RngState rng(12345ULL);
  raft::random::uniform(handle, rng, X, n_rows * dim, float(-1), float(1));
  raft::random::uniform(handle, rng, centroids, n_clusters * dim, float(-1), float(1));
  raft::resource::sync_stream(handle);
}

}  // namespace

static void BM_ClusterAssignment_BruteForce(benchmark::State& state)
{
  int64_t n_rows     = static_cast<int64_t>(state.range(0));
  int64_t n_clusters = static_cast<int64_t>(state.range(1));
  int64_t dim        = static_cast<int64_t>(state.range(2));

  raft::device_resources handle;
  rmm::device_uvector<float> X(static_cast<size_t>(n_rows) * static_cast<size_t>(dim),
                               raft::resource::get_cuda_stream(handle));
  rmm::device_uvector<float> centroids(static_cast<size_t>(n_clusters) * static_cast<size_t>(dim),
                                       raft::resource::get_cuda_stream(handle));
  rmm::device_uvector<uint32_t> labels(static_cast<size_t>(n_rows),
                                       raft::resource::get_cuda_stream(handle));

  init_random_data(handle, X.data(), n_rows, dim, centroids.data(), n_clusters);

  cuvs::cluster::kmeans::balanced_params params;
  params.metric = cuvs::distance::DistanceType::L2Expanded;

  auto X_view = raft::make_device_matrix_view<const float, int64_t>(X.data(), n_rows, dim);
  auto centers_view =
    raft::make_device_matrix_view<const float, int64_t>(centroids.data(), n_clusters, dim);
  auto labels_view = raft::make_device_vector_view<uint32_t, int64_t>(labels.data(), n_rows);

  for (auto _ : state) {
    predict(handle, params, X_view, centers_view, labels_view);
    raft::resource::sync_stream(handle);
  }
  state.SetItemsProcessed(state.iterations() * n_rows);
}

static void BM_ClusterAssignment_CAGRA(benchmark::State& state)
{
  int64_t n_rows     = static_cast<int64_t>(state.range(0));
  int64_t n_clusters = static_cast<int64_t>(state.range(1));
  int64_t dim        = static_cast<int64_t>(state.range(2));

  raft::device_resources handle;
  rmm::device_uvector<float> X(static_cast<size_t>(n_rows) * static_cast<size_t>(dim),
                               raft::resource::get_cuda_stream(handle));
  rmm::device_uvector<float> centroids(static_cast<size_t>(n_clusters) * static_cast<size_t>(dim),
                                       raft::resource::get_cuda_stream(handle));
  rmm::device_uvector<uint32_t> labels(static_cast<size_t>(n_rows),
                                       raft::resource::get_cuda_stream(handle));

  init_random_data(handle, X.data(), n_rows, dim, centroids.data(), n_clusters);

  cuvs::cluster::kmeans::balanced_params params;
  params.metric = cuvs::distance::DistanceType::L2Expanded;

  // Same timing as assign_nearest_centroid_cagra_with_index_reuse with rebuild=true each iteration.
  // float X/centroids only.
  std::optional<cuvs::neighbors::cagra::index<float, uint32_t>> cagra_index_opt;

  for (auto _ : state) {
    cuvs::cluster::kmeans::detail::assign_nearest_centroid_cagra_with_index_reuse<int64_t,
                                                                                  uint32_t>(
      handle,
      params,
      centroids.data(),
      n_clusters,
      dim,
      X.data(),
      n_rows,
      labels.data(),
      &cagra_index_opt,
      true);
    raft::resource::sync_stream(handle);
  }
  state.SetItemsProcessed(state.iterations() * n_rows);
}

// N = vectors to assign, K = number of clusters, D = dimension
// Small: 10K vectors, 1K clusters, 128 dim
BENCHMARK(BM_ClusterAssignment_BruteForce)
  ->Args({10000, 1000, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();
BENCHMARK(BM_ClusterAssignment_CAGRA)
  ->Args({10000, 1000, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

// Medium: 100K vectors, 4K clusters
BENCHMARK(BM_ClusterAssignment_BruteForce)
  ->Args({100000, 4000, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();
BENCHMARK(BM_ClusterAssignment_CAGRA)
  ->Args({100000, 4000, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

// Large K: 100K vectors, 16K clusters (brute force starts to hurt)
BENCHMARK(BM_ClusterAssignment_BruteForce)
  ->Args({100000, 16000, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();
BENCHMARK(BM_ClusterAssignment_CAGRA)
  ->Args({100000, 16000, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

// Very large K: 500K vectors, 64K clusters
BENCHMARK(BM_ClusterAssignment_BruteForce)
  ->Args({500000, 65536, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();
BENCHMARK(BM_ClusterAssignment_CAGRA)
  ->Args({500000, 65536, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

// Larger N: amortize CAGRA build over more queries
// 1M vectors, 4K clusters
BENCHMARK(BM_ClusterAssignment_BruteForce)
  ->Args({1000000, 4000, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();
BENCHMARK(BM_ClusterAssignment_CAGRA)
  ->Args({1000000, 4000, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

// 1M vectors, 16K clusters
BENCHMARK(BM_ClusterAssignment_BruteForce)
  ->Args({1000000, 16000, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();
BENCHMARK(BM_ClusterAssignment_CAGRA)
  ->Args({1000000, 16000, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

// 1M vectors, 64K clusters
BENCHMARK(BM_ClusterAssignment_BruteForce)
  ->Args({1000000, 65536, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();
BENCHMARK(BM_ClusterAssignment_CAGRA)
  ->Args({1000000, 65536, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

// 2M vectors, 16K clusters
BENCHMARK(BM_ClusterAssignment_BruteForce)
  ->Args({2000000, 16000, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();
BENCHMARK(BM_ClusterAssignment_CAGRA)
  ->Args({2000000, 16000, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

// 2M vectors, 64K clusters
BENCHMARK(BM_ClusterAssignment_BruteForce)
  ->Args({2000000, 65536, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();
BENCHMARK(BM_ClusterAssignment_CAGRA)
  ->Args({2000000, 65536, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

// 5M vectors, 16K clusters
BENCHMARK(BM_ClusterAssignment_BruteForce)
  ->Args({5000000, 16000, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();
BENCHMARK(BM_ClusterAssignment_CAGRA)
  ->Args({5000000, 16000, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

// 5M vectors, 64K clusters
BENCHMARK(BM_ClusterAssignment_BruteForce)
  ->Args({5000000, 65536, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();
BENCHMARK(BM_ClusterAssignment_CAGRA)
  ->Args({5000000, 65536, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

// Hundreds of thousands of centroids (K = 100K, 200K, 500K, 1M)
// 1M vectors, 100K clusters
BENCHMARK(BM_ClusterAssignment_BruteForce)
  ->Args({1000000, 100000, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();
BENCHMARK(BM_ClusterAssignment_CAGRA)
  ->Args({1000000, 100000, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

// 2M vectors, 100K clusters
BENCHMARK(BM_ClusterAssignment_BruteForce)
  ->Args({2000000, 100000, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();
BENCHMARK(BM_ClusterAssignment_CAGRA)
  ->Args({2000000, 100000, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

// 1M vectors, 200K clusters
BENCHMARK(BM_ClusterAssignment_BruteForce)
  ->Args({1000000, 200000, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();
BENCHMARK(BM_ClusterAssignment_CAGRA)
  ->Args({1000000, 200000, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

// 1M vectors, 500K clusters (~2 vectors per cluster)
BENCHMARK(BM_ClusterAssignment_BruteForce)
  ->Args({1000000, 500000, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();
BENCHMARK(BM_ClusterAssignment_CAGRA)
  ->Args({1000000, 500000, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

// 1M clusters with N > K (realistic: many vectors per cluster)
// 2M vectors, 1M clusters (~2 per cluster)
BENCHMARK(BM_ClusterAssignment_BruteForce)
  ->Args({2000000, 1000000, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();
BENCHMARK(BM_ClusterAssignment_CAGRA)
  ->Args({2000000, 1000000, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

// 5M vectors, 1M clusters (~5 per cluster)
BENCHMARK(BM_ClusterAssignment_BruteForce)
  ->Args({5000000, 1000000, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();
BENCHMARK(BM_ClusterAssignment_CAGRA)
  ->Args({5000000, 1000000, 128})
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

BENCHMARK_MAIN();

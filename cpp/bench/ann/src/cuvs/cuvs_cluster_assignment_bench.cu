/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Search-only variant: builds the CAGRA index once (outside the timed loop) and only times
// repeated searches against that fixed index. Isolates steady-state search cost from the
// one-time graph-build cost that BM_ClusterAssignment_CAGRA pays on every iteration -- this
// matches the real k-means fit loop when ann_rebuild_interval > 1 amortizes the build.
static void BM_ClusterAssignment_CAGRA_SearchOnly(benchmark::State& state)
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

  auto X_view =
    raft::make_device_matrix_view<const float, int64_t, raft::row_major>(X.data(), n_rows, dim);
  auto centers_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    centroids.data(), n_clusters, dim);

  auto cagra_index =
    cuvs::cluster::kmeans::detail::build_cagra_index_for_centroids(handle, params, centers_view);
  auto search_params = cuvs::cluster::kmeans::detail::default_cagra_centroid_search_params();

  for (auto _ : state) {
    cuvs::cluster::kmeans::detail::assign_nearest_centroid_cagra(
      handle, search_params, cagra_index, X_view, labels.data(), n_rows);
    raft::resource::sync_stream(handle);
  }
  state.SetItemsProcessed(state.iterations() * n_rows);
}

// Amortized-interval variant: times one full rebuild cycle -- build the CAGRA index once, then
// run `interval` search calls back-to-back against it -- all inside the timed region, matching
// how a real k-means fit loop uses the index across `ann_rebuild_interval` E-steps (rebuild once,
// reuse for the rest of the interval). Unlike deriving this by formula from separately-measured
// build and search numbers, this actually executes the searches back-to-back so any real GPU
// effects (caching, thermal state, sustained-load behavior) show up in the measurement.
static void BM_ClusterAssignment_CAGRA_AmortizedInterval(benchmark::State& state)
{
  int64_t n_rows     = static_cast<int64_t>(state.range(0));
  int64_t n_clusters = static_cast<int64_t>(state.range(1));
  int64_t dim        = static_cast<int64_t>(state.range(2));
  int64_t interval   = static_cast<int64_t>(state.range(3));

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

  auto X_view =
    raft::make_device_matrix_view<const float, int64_t, raft::row_major>(X.data(), n_rows, dim);
  auto centers_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    centroids.data(), n_clusters, dim);
  auto search_params = cuvs::cluster::kmeans::detail::default_cagra_centroid_search_params();

  for (auto _ : state) {
    auto cagra_index =
      cuvs::cluster::kmeans::detail::build_cagra_index_for_centroids(handle, params, centers_view);
    for (int64_t s = 0; s < interval; ++s) {
      cuvs::cluster::kmeans::detail::assign_nearest_centroid_cagra(
        handle, search_params, cagra_index, X_view, labels.data(), n_rows);
    }
    raft::resource::sync_stream(handle);
  }
  state.SetItemsProcessed(state.iterations() * n_rows * interval);
}

// Controlled sweep: hold points-per-cluster fixed at kPointsPerCluster (N = kPointsPerCluster * K)
// so K is the only independent variable. This isolates the effect of K on the brute-force-vs-CAGRA
// crossover; letting both N and K vary independently (as the old hand-picked Args list did)
// confounds the two and makes the crossover point ill-defined.
constexpr int64_t kPointsPerCluster = 5;
constexpr int64_t kDim              = 128;

// clang-format off
constexpr int64_t kClusterCounts[] = {
  1000, 2000, 4000, 8000, 16000, 32000, 65536, 131072, 262144, 500000, 1000000
};
// clang-format on

static void RegisterConstantRatioSweep()
{
  for (int64_t k : kClusterCounts) {
    int64_t n = kPointsPerCluster * k;
    benchmark::RegisterBenchmark("BM_ClusterAssignment_BruteForce", BM_ClusterAssignment_BruteForce)
      ->Args({n, k, kDim})
      ->Unit(benchmark::kMillisecond)
      ->UseRealTime();
    benchmark::RegisterBenchmark("BM_ClusterAssignment_CAGRA", BM_ClusterAssignment_CAGRA)
      ->Args({n, k, kDim})
      ->Unit(benchmark::kMillisecond)
      ->UseRealTime();
  }
}

// Second controlled sweep: hold N fixed and vary K alone (same kClusterCounts list, so the two
// sweeps are directly comparable). This isolates the effect of K on the crossover from the effect
// of the N/K ratio tested above -- it answers "at a fixed dataset size, does ANN help more as the
// number of clusters grows?" rather than "does ANN help more as both grow together?".
// kFixedN is chosen to not collide with any kPointsPerCluster * K value from the sweep above.
constexpr int64_t kFixedN = 2000000;

static void RegisterFixedNVaryKSweep()
{
  for (int64_t k : kClusterCounts) {
    benchmark::RegisterBenchmark("BM_ClusterAssignment_BruteForce", BM_ClusterAssignment_BruteForce)
      ->Args({kFixedN, k, kDim})
      ->Unit(benchmark::kMillisecond)
      ->UseRealTime();
    benchmark::RegisterBenchmark("BM_ClusterAssignment_CAGRA", BM_ClusterAssignment_CAGRA)
      ->Args({kFixedN, k, kDim})
      ->Unit(benchmark::kMillisecond)
      ->UseRealTime();
  }
}

// Fixed-K, vary-N sweep: hold K fixed (below the fixed-N crossover found earlier) and grow N.
// Complements RegisterFixedNVaryKSweep -- answers whether growing N alone (holding K fixed) can
// also flip brute-force-vs-CAGRA, by amortizing CAGRA's one-time graph-build cost over more
// queries, or whether the crossover is driven by K alone regardless of N.
constexpr int64_t kFixedKForNSweep = 65536;
// clang-format off
constexpr int64_t kNValuesForFixedK[] = {200000, 655360, 1310720, 5000000, 10000000};
// clang-format on

static void RegisterFixedKVaryNSweep()
{
  for (int64_t n : kNValuesForFixedK) {
    benchmark::RegisterBenchmark("BM_ClusterAssignment_BruteForce", BM_ClusterAssignment_BruteForce)
      ->Args({n, kFixedKForNSweep, kDim})
      ->Unit(benchmark::kMillisecond)
      ->UseRealTime();
    benchmark::RegisterBenchmark("BM_ClusterAssignment_CAGRA", BM_ClusterAssignment_CAGRA)
      ->Args({n, kFixedKForNSweep, kDim})
      ->Unit(benchmark::kMillisecond)
      ->UseRealTime();
  }
}

// Coarse (N, K) grid: combined with the two sweeps above, gives enough points spread across the
// (N, K) plane to sketch a crossover boundary instead of inferring it from 1D slices alone.
// clang-format off
constexpr int64_t kGridN[] = {500000, 2000000, 5000000};
constexpr int64_t kGridK[] = {4000, 65536, 500000};
// clang-format on

static void RegisterGridSweep()
{
  for (int64_t n : kGridN) {
    for (int64_t k : kGridK) {
      benchmark::RegisterBenchmark("BM_ClusterAssignment_BruteForce",
                                   BM_ClusterAssignment_BruteForce)
        ->Args({n, k, kDim})
        ->Unit(benchmark::kMillisecond)
        ->UseRealTime();
      benchmark::RegisterBenchmark("BM_ClusterAssignment_CAGRA", BM_ClusterAssignment_CAGRA)
        ->Args({n, k, kDim})
        ->Unit(benchmark::kMillisecond)
        ->UseRealTime();
    }
  }
}

// Build-vs-search breakdown: CAGRA search-only cost (index built once, outside the timed loop)
// across the same K sweep as RegisterFixedNVaryKSweep, so it lines up with that sweep's brute
// force and always-rebuild CAGRA points on the same (N, K) axis for direct comparison.
static void RegisterCagraSearchOnlySweep()
{
  for (int64_t k : kClusterCounts) {
    benchmark::RegisterBenchmark("BM_ClusterAssignment_CAGRA_SearchOnly",
                                 BM_ClusterAssignment_CAGRA_SearchOnly)
      ->Args({kFixedN, k, kDim})
      ->Unit(benchmark::kMillisecond)
      ->UseRealTime();
  }
}

// Dimension sensitivity: fixed N and K (near the fixed-N crossover found earlier), dim varies
// across realistic embedding sizes. Checks whether the crossover K shifts materially with dim,
// since dim=128 was the only value tested elsewhere in this file.
constexpr int64_t kFixedNForDimSweep = 2000000;
constexpr int64_t kFixedKForDimSweep = 131072;
// clang-format off
constexpr int64_t kDimValues[] = {64, 128, 256, 512, 1024};
// clang-format on

static void RegisterDimSensitivitySweep()
{
  for (int64_t dim : kDimValues) {
    benchmark::RegisterBenchmark("BM_ClusterAssignment_BruteForce", BM_ClusterAssignment_BruteForce)
      ->Args({kFixedNForDimSweep, kFixedKForDimSweep, dim})
      ->Unit(benchmark::kMillisecond)
      ->UseRealTime();
    benchmark::RegisterBenchmark("BM_ClusterAssignment_CAGRA", BM_ClusterAssignment_CAGRA)
      ->Args({kFixedNForDimSweep, kFixedKForDimSweep, dim})
      ->Unit(benchmark::kMillisecond)
      ->UseRealTime();
  }
}

// Empirical build-vs-search breakdown: same representative K subset as the derived version, but
// each (K, interval) pair is a real measurement of "build once, search `interval` times back to
// back" rather than a formula applied to separately-measured build/search numbers.
// clang-format off
constexpr int64_t kAmortizedIntervalKs[] = {1000, 65536, 131072, 262144, 500000, 1000000};
constexpr int64_t kAmortizedIntervals[]  = {1, 2, 5, 10};
// clang-format on

static void RegisterCagraAmortizedIntervalSweep()
{
  for (int64_t k : kAmortizedIntervalKs) {
    for (int64_t interval : kAmortizedIntervals) {
      benchmark::RegisterBenchmark("BM_ClusterAssignment_CAGRA_AmortizedInterval",
                                   BM_ClusterAssignment_CAGRA_AmortizedInterval)
        ->Args({kFixedN, k, kDim, interval})
        ->Unit(benchmark::kMillisecond)
        ->UseRealTime();
    }
  }
}

int main(int argc, char** argv)
{
  RegisterConstantRatioSweep();
  RegisterFixedNVaryKSweep();
  RegisterFixedKVaryNSweep();
  RegisterGridSweep();
  RegisterCagraSearchOnlySweep();
  RegisterDimSensitivitySweep();
  RegisterCagraAmortizedIntervalSweep();
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) { return 1; }
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}

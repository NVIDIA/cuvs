/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tuning benchmark for balanced k-means fit E-step ANN rebuild interval (`ann_rebuild_interval`).
 *
 * Three modes:
 *  - Brute: `use_ann_for_build_fit = false`
 *  - CAGRA rebuild every EM iteration: `ann_rebuild_interval = 1`
 *  - CAGRA index reuse: `ann_rebuild_interval = R` (R > 1)
 *
 * For each case reports fit time and accuracy vs a brute-force reference fit on the same data:
 *  - label_match_pct: % of points assigned to the same cluster (via predict on final centroids)
 *  - centroid_mean_l2_drift: mean per-centroid L2 distance between final centroids
 *
 * Label match and centroid drift are computed once after the full fit (all n_iters),
 * comparing final centroids and final predict labels vs brute — not per EM iteration.
 */
#include <benchmark/benchmark.h>

#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/random/rng.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr uint32_t kFitIters = 3;

struct BenchRanges {
  int64_t n_rows;
  int64_t n_clusters;
  int64_t dim;
};

BenchRanges read_ranges(benchmark::State const& state)
{
  return BenchRanges{static_cast<int64_t>(state.range(0)),
                     static_cast<int64_t>(state.range(1)),
                     static_cast<int64_t>(state.range(2))};
}

void init_random_dataset(raft::resources const& handle, float* data, int64_t n_rows, int64_t dim)
{
  raft::random::RngState rng(12345ULL);
  raft::random::uniform(handle, rng, data, n_rows * dim, float(-1), float(1));
  raft::resource::sync_stream(handle);
}

cuvs::cluster::kmeans::balanced_params make_fit_params(bool use_ann, uint32_t ann_rebuild_interval)
{
  cuvs::cluster::kmeans::balanced_params params;
  params.n_iters               = kFitIters;
  params.metric                = cuvs::distance::DistanceType::L2Expanded;
  params.use_ann_for_build_fit = use_ann;
  params.ann_rebuild_interval  = ann_rebuild_interval;
  return params;
}

void run_fit(raft::resources const& handle,
             cuvs::cluster::kmeans::balanced_params const& params,
             raft::device_matrix_view<const float, int64_t> dataset_view,
             raft::device_matrix_view<float, int64_t> centroids_view)
{
  cuvs::cluster::kmeans::fit(handle, params, dataset_view, centroids_view);
  raft::resource::sync_stream(handle);
}

void run_predict(raft::resources const& handle,
                 cuvs::cluster::kmeans::balanced_params const& params,
                 raft::device_matrix_view<const float, int64_t> dataset_view,
                 raft::device_matrix_view<const float, int64_t> centroids_view,
                 raft::device_vector_view<uint32_t, int64_t> labels_view)
{
  cuvs::cluster::kmeans::predict(handle, params, dataset_view, centroids_view, labels_view);
  raft::resource::sync_stream(handle);
}

std::vector<uint32_t> copy_labels_to_host(raft::resources const& handle,
                                          uint32_t const* labels,
                                          int64_t n_rows)
{
  std::vector<uint32_t> labels_host(static_cast<size_t>(n_rows));
  raft::copy(labels_host.data(),
             labels,
             static_cast<size_t>(n_rows),
             raft::resource::get_cuda_stream(handle));
  raft::resource::sync_stream(handle);
  return labels_host;
}

std::vector<float> copy_centroids_to_host(raft::resources const& handle,
                                          float const* centroids,
                                          int64_t n_clusters,
                                          int64_t dim)
{
  const size_t n_elems = static_cast<size_t>(n_clusters) * static_cast<size_t>(dim);
  std::vector<float> centroids_host(n_elems);
  raft::copy(centroids_host.data(), centroids, n_elems, raft::resource::get_cuda_stream(handle));
  raft::resource::sync_stream(handle);
  return centroids_host;
}

double label_match_pct(const std::vector<uint32_t>& ref_labels,
                       const std::vector<uint32_t>& test_labels)
{
  const size_t n   = ref_labels.size();
  uint64_t matches = 0;
  for (size_t i = 0; i < n; ++i) {
    if (ref_labels[i] == test_labels[i]) { ++matches; }
  }
  return 100.0 * static_cast<double>(matches) / static_cast<double>(n);
}

double centroid_mean_l2_drift(const std::vector<float>& ref_centroids,
                              const std::vector<float>& test_centroids,
                              int64_t n_clusters,
                              int64_t dim)
{
  double sum_l2 = 0.0;
  for (int64_t c = 0; c < n_clusters; ++c) {
    double sq = 0.0;
    for (int64_t d = 0; d < dim; ++d) {
      const size_t idx = static_cast<size_t>(c) * static_cast<size_t>(dim) + static_cast<size_t>(d);
      const double diff =
        static_cast<double>(ref_centroids[idx]) - static_cast<double>(test_centroids[idx]);
      sq += diff * diff;
    }
    sum_l2 += std::sqrt(sq);
  }
  return sum_l2 / static_cast<double>(n_clusters);
}

struct FitFixture {
  raft::device_resources handle;
  rmm::device_uvector<float> dataset;
  rmm::device_uvector<float> centroids;
  rmm::device_uvector<uint32_t> labels;
  raft::device_matrix_view<const float, int64_t> dataset_view;
  raft::device_matrix_view<float, int64_t> centroids_view;
  raft::device_vector_view<uint32_t, int64_t> labels_view;

  explicit FitFixture(BenchRanges const& ranges)
    : handle{},
      dataset(static_cast<size_t>(ranges.n_rows) * static_cast<size_t>(ranges.dim),
              raft::resource::get_cuda_stream(handle)),
      centroids(static_cast<size_t>(ranges.n_clusters) * static_cast<size_t>(ranges.dim),
                raft::resource::get_cuda_stream(handle)),
      labels(static_cast<size_t>(ranges.n_rows), raft::resource::get_cuda_stream(handle))
  {
    init_random_dataset(handle, dataset.data(), ranges.n_rows, ranges.dim);
    dataset_view = raft::make_device_matrix_view<const float, int64_t>(
      dataset.data(), ranges.n_rows, ranges.dim);
    centroids_view = raft::make_device_matrix_view<float, int64_t>(
      centroids.data(), ranges.n_clusters, ranges.dim);
    labels_view = raft::make_device_vector_view<uint32_t, int64_t>(labels.data(), ranges.n_rows);
  }
};

struct BruteReference {
  std::vector<uint32_t> labels;
  std::vector<float> centroids;
  double fit_ms = 0.0;
};

BruteReference compute_brute_reference(FitFixture& fixture, BenchRanges const& ranges)
{
  auto brute_params = make_fit_params(false, 1);
  auto ref_centroids =
    raft::make_device_matrix<float, int64_t>(fixture.handle, ranges.n_clusters, ranges.dim);

  const auto start = std::chrono::steady_clock::now();
  run_fit(fixture.handle, brute_params, fixture.dataset_view, ref_centroids.view());
  const auto end      = std::chrono::steady_clock::now();
  const double fit_ms = 1e-6 * std::chrono::duration<double, std::nano>(end - start).count();

  run_predict(fixture.handle,
              brute_params,
              fixture.dataset_view,
              raft::make_const_mdspan(ref_centroids.view()),
              fixture.labels_view);

  BruteReference ref;
  ref.fit_ms    = fit_ms;
  ref.labels    = copy_labels_to_host(fixture.handle, fixture.labels.data(), ranges.n_rows);
  ref.centroids = copy_centroids_to_host(
    fixture.handle, ref_centroids.data_handle(), ranges.n_clusters, ranges.dim);
  return ref;
}

static void BM_FitAnnTuning_Brute(benchmark::State& state)
{
  auto ranges = read_ranges(state);
  FitFixture fixture(ranges);
  auto params = make_fit_params(false, 1);

  for (auto _ : state) {
    run_fit(fixture.handle, params, fixture.dataset_view, fixture.centroids_view);
  }

  state.SetItemsProcessed(state.iterations() * ranges.n_rows);
  state.counters["label_match_pct"]        = 100.0;
  state.counters["centroid_mean_l2_drift"] = 0.0;
}

static void BM_FitAnnTuning_CAGRA(benchmark::State& state)
{
  auto ranges                         = read_ranges(state);
  const uint32_t ann_rebuild_interval = static_cast<uint32_t>(state.range(3));
  FitFixture fixture(ranges);
  auto ann_params = make_fit_params(true, ann_rebuild_interval);

  double last_label_match      = 100.0;
  double last_centroid_drift   = 0.0;
  double last_speedup_vs_brute = 0.0;

  for (auto _ : state) {
    state.PauseTiming();
    const BruteReference brute_ref = compute_brute_reference(fixture, ranges);
    state.ResumeTiming();

    const auto ann_start = std::chrono::steady_clock::now();
    run_fit(fixture.handle, ann_params, fixture.dataset_view, fixture.centroids_view);
    const auto ann_end = std::chrono::steady_clock::now();
    const double ann_fit_ms =
      1e-6 * std::chrono::duration<double, std::nano>(ann_end - ann_start).count();

    state.PauseTiming();
    run_predict(fixture.handle,
                ann_params,
                fixture.dataset_view,
                raft::make_const_mdspan(fixture.centroids_view),
                fixture.labels_view);

    const auto ann_labels =
      copy_labels_to_host(fixture.handle, fixture.labels.data(), ranges.n_rows);
    const auto ann_centroids = copy_centroids_to_host(
      fixture.handle, fixture.centroids.data(), ranges.n_clusters, ranges.dim);

    last_label_match = label_match_pct(brute_ref.labels, ann_labels);
    last_centroid_drift =
      centroid_mean_l2_drift(brute_ref.centroids, ann_centroids, ranges.n_clusters, ranges.dim);

    last_speedup_vs_brute = ann_fit_ms > 0.0 ? brute_ref.fit_ms / ann_fit_ms : 0.0;
    state.ResumeTiming();
  }

  state.SetItemsProcessed(state.iterations() * ranges.n_rows);
  state.counters["label_match_pct"]        = last_label_match;
  state.counters["centroid_mean_l2_drift"] = last_centroid_drift;
  state.counters["speedup_vs_brute"]       = last_speedup_vs_brute;
}

constexpr int64_t kDim = 128;

#define FIT_ANN_TUNING_BENCH_ARGS(BM)            \
  BENCHMARK(BM)                                  \
    ->Args({327680, 65536, kDim})                \
    ->Unit(benchmark::kMillisecond)              \
    ->UseRealTime()                              \
    ->ArgNames({"n_vectors", "n_lists", "dim"}); \
  BENCHMARK(BM)                                  \
    ->Args({1000000, 200000, kDim})              \
    ->Unit(benchmark::kMillisecond)              \
    ->UseRealTime()                              \
    ->ArgNames({"n_vectors", "n_lists", "dim"}); \
  BENCHMARK(BM)                                  \
    ->Args({1500000, 300000, kDim})              \
    ->Unit(benchmark::kMillisecond)              \
    ->UseRealTime()                              \
    ->ArgNames({"n_vectors", "n_lists", "dim"}); \
  BENCHMARK(BM)                                  \
    ->Args({1750000, 350000, kDim})              \
    ->Unit(benchmark::kMillisecond)              \
    ->UseRealTime()                              \
    ->ArgNames({"n_vectors", "n_lists", "dim"}); \
  BENCHMARK(BM)                                  \
    ->Args({2000000, 400000, kDim})              \
    ->Unit(benchmark::kMillisecond)              \
    ->UseRealTime()                              \
    ->ArgNames({"n_vectors", "n_lists", "dim"}); \
  BENCHMARK(BM)                                  \
    ->Args({3000000, 600000, kDim})              \
    ->Unit(benchmark::kMillisecond)              \
    ->UseRealTime()                              \
    ->ArgNames({"n_vectors", "n_lists", "dim"}); \
  BENCHMARK(BM)                                  \
    ->Args({4000000, 800000, kDim})              \
    ->Unit(benchmark::kMillisecond)              \
    ->UseRealTime()                              \
    ->ArgNames({"n_vectors", "n_lists", "dim"}); \
  BENCHMARK(BM)                                  \
    ->Args({5000000, 1000000, kDim})             \
    ->Unit(benchmark::kMillisecond)              \
    ->UseRealTime()                              \
    ->ArgNames({"n_vectors", "n_lists", "dim"});

#define FIT_ANN_TUNING_CAGRA_ARGS(BM)                                    \
  BENCHMARK(BM)                                                          \
    ->ArgsProduct({{327680}, {65536}, {kDim}, {1, 2, 3, 5, 10}})         \
    ->Unit(benchmark::kMillisecond)                                      \
    ->UseRealTime()                                                      \
    ->ArgNames({"n_vectors", "n_lists", "dim", "ann_rebuild_interval"}); \
  BENCHMARK(BM)                                                          \
    ->ArgsProduct({{1000000}, {200000}, {kDim}, {1, 2, 3, 5, 10}})       \
    ->Unit(benchmark::kMillisecond)                                      \
    ->UseRealTime()                                                      \
    ->ArgNames({"n_vectors", "n_lists", "dim", "ann_rebuild_interval"}); \
  BENCHMARK(BM)                                                          \
    ->ArgsProduct({{1500000}, {300000}, {kDim}, {1, 2, 3, 5, 10}})       \
    ->Unit(benchmark::kMillisecond)                                      \
    ->UseRealTime()                                                      \
    ->ArgNames({"n_vectors", "n_lists", "dim", "ann_rebuild_interval"}); \
  BENCHMARK(BM)                                                          \
    ->ArgsProduct({{1750000}, {350000}, {kDim}, {1, 2, 3, 5, 10}})       \
    ->Unit(benchmark::kMillisecond)                                      \
    ->UseRealTime()                                                      \
    ->ArgNames({"n_vectors", "n_lists", "dim", "ann_rebuild_interval"}); \
  BENCHMARK(BM)                                                          \
    ->ArgsProduct({{2000000}, {400000}, {kDim}, {1, 2, 3, 5, 10}})       \
    ->Unit(benchmark::kMillisecond)                                      \
    ->UseRealTime()                                                      \
    ->ArgNames({"n_vectors", "n_lists", "dim", "ann_rebuild_interval"}); \
  BENCHMARK(BM)                                                          \
    ->ArgsProduct({{3000000}, {600000}, {kDim}, {1, 2, 3, 5, 10}})       \
    ->Unit(benchmark::kMillisecond)                                      \
    ->UseRealTime()                                                      \
    ->ArgNames({"n_vectors", "n_lists", "dim", "ann_rebuild_interval"}); \
  BENCHMARK(BM)                                                          \
    ->ArgsProduct({{4000000}, {800000}, {kDim}, {1, 2, 3, 5, 10}})       \
    ->Unit(benchmark::kMillisecond)                                      \
    ->UseRealTime()                                                      \
    ->ArgNames({"n_vectors", "n_lists", "dim", "ann_rebuild_interval"}); \
  BENCHMARK(BM)                                                          \
    ->ArgsProduct({{5000000}, {1000000}, {kDim}, {1, 2, 3, 5, 10}})      \
    ->Unit(benchmark::kMillisecond)                                      \
    ->UseRealTime()                                                      \
    ->ArgNames({"n_vectors", "n_lists", "dim", "ann_rebuild_interval"});

FIT_ANN_TUNING_BENCH_ARGS(BM_FitAnnTuning_Brute);
FIT_ANN_TUNING_CAGRA_ARGS(BM_FitAnnTuning_CAGRA);

}  // namespace

BENCHMARK_MAIN();

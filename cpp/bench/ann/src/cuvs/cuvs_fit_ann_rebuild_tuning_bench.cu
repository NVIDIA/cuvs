/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tuning benchmark for the ANN assignment in Lloyd's k-means fit (cuvs::cluster::kmeans::fit with
 * kmeans::params::use_ann_for_assignment and `ann_itopk_size`).
 *
 *  - BM_LloydFit_Brute: use_ann_for_assignment = false (the baseline)
 *  - BM_LloydFit_ANN:   use_ann_for_assignment = true, swept over ann_itopk_size (the index is
 *                       rebuilt every iteration)
 *
 * Both fits use the same data, the same seed, init = Random and a fixed iteration count
 * (tol is tiny so no early stop), so the timings compare the same amount of work. Reported
 * against the brute-force fit on the same data:
 *  - speedup_vs_brute:       brute fit time / ANN fit time
 *  - inertia_ratio_vs_brute: exact final inertia of the ANN fit / that of the brute fit
 *                            (1.0 = same quality, > 1 = worse)
 *  - label_match_pct:        % of points assigned to the same cluster id by predict on the two
 *                            sets of final centroids. This compares cluster ids index-by-index
 *                            between two independent fits, so it collapses once the trajectories
 *                            diverge even when both fits are equally good; use the inertia ratio
 *                            as the quality measure.
 *  - ann_1nn_agreement_pct_itopk{32,64,256}: % of points for which a CAGRA 1-NN over the brute
 *                            fit's (fixed) final centroids returns the exact nearest centroid.
 */
#include <benchmark/benchmark.h>

#include "bench_dataset.hpp"

#include <cluster/detail/kmeans_ann_assignment_core.cuh>
#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <tuple>
#include <vector>

namespace {

constexpr int kFitIters = 20;

struct BenchRanges {
  int64_t n_rows;
  int64_t n_clusters;
  int64_t dim;
};

BenchRanges read_ranges(benchmark::State const& state)
{
  return {static_cast<int64_t>(state.range(0)),
          static_cast<int64_t>(state.range(1)),
          static_cast<int64_t>(state.range(2))};
}

cuvs::cluster::kmeans::params make_fit_params(int64_t n_clusters, bool use_ann, size_t itopk = 64)
{
  cuvs::cluster::kmeans::params params;
  params.n_clusters             = static_cast<int>(n_clusters);
  params.metric                 = cuvs::distance::DistanceType::L2Expanded;
  params.init                   = cuvs::cluster::kmeans::params::Random;
  params.max_iter               = kFitIters;
  params.tol                    = 1e-12;  // run all kFitIters iterations
  params.n_init                 = 1;
  params.rng_state              = raft::random::RngState(42);
  params.use_ann_for_assignment = use_ann;
  params.ann_itopk_size         = itopk;
  return params;
}

struct FitFixture {
  raft::device_resources handle;
  rmm::device_uvector<float> dataset;
  rmm::device_uvector<float> centroids;
  rmm::device_uvector<int64_t> labels;
  BenchRanges ranges;

  explicit FitFixture(BenchRanges const& r)
    : handle{},
      dataset(static_cast<size_t>(r.n_rows) * r.dim, raft::resource::get_cuda_stream(handle)),
      centroids(static_cast<size_t>(r.n_clusters) * r.dim, raft::resource::get_cuda_stream(handle)),
      labels(static_cast<size_t>(r.n_rows), raft::resource::get_cuda_stream(handle)),
      ranges(r)
  {
    bench_dataset::load_data(handle, dataset.data(), r.n_rows, r.dim);
  }

  auto X() const
  {
    return raft::make_device_matrix_view<const float, int64_t>(
      dataset.data(), ranges.n_rows, ranges.dim);
  }
  auto C(float* c) const
  {
    return raft::make_device_matrix_view<float, int64_t>(c, ranges.n_clusters, ranges.dim);
  }

  // Runs fit and returns {wall ms, exact inertia, iterations run}; final centroids land in `out`.
  std::tuple<double, double, int64_t> fit(cuvs::cluster::kmeans::params const& params, float* out)
  {
    float inertia  = 0;
    int64_t n_iter = 0;
    const auto t0  = std::chrono::steady_clock::now();
    cuvs::cluster::kmeans::fit(handle,
                               params,
                               X(),
                               std::nullopt,
                               C(out),
                               raft::make_host_scalar_view(&inertia),
                               raft::make_host_scalar_view(&n_iter));
    raft::resource::sync_stream(handle);
    const auto t1 = std::chrono::steady_clock::now();
    return {1e-6 * std::chrono::duration<double, std::nano>(t1 - t0).count(), inertia, n_iter};
  }

  // Brute-force predict labels for the given centroids, copied to the host.
  std::vector<int64_t> predict_labels(float const* centroids_dev)
  {
    float inertia = 0;
    cuvs::cluster::kmeans::predict(
      handle,
      make_fit_params(ranges.n_clusters, false),
      X(),
      std::nullopt,
      raft::make_device_matrix_view<const float, int64_t>(
        centroids_dev, ranges.n_clusters, ranges.dim),
      raft::make_device_vector_view<int64_t, int64_t>(labels.data(), ranges.n_rows),
      false,
      raft::make_host_scalar_view(&inertia));
    return labels_to_host();
  }

  std::vector<int64_t> labels_to_host()
  {
    std::vector<int64_t> h(static_cast<size_t>(ranges.n_rows));
    raft::copy(h.data(), labels.data(), h.size(), raft::resource::get_cuda_stream(handle));
    raft::resource::sync_stream(handle);
    return h;
  }
};

double label_match_pct(std::vector<int64_t> const& a, std::vector<int64_t> const& b)
{
  uint64_t same = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    same += (a[i] == b[i]);
  }
  return 100.0 * static_cast<double>(same) / static_cast<double>(a.size());
}

// % of points whose CAGRA 1-NN over `centroids_dev` (fixed, so no staleness) equals `exact`.
double ann_1nn_agreement_pct(FitFixture& f,
                             float const* centroids_dev,
                             std::vector<int64_t> const& exact,
                             size_t itopk)
{
  auto params = make_fit_params(f.ranges.n_clusters, true, itopk);
  cuvs::distance::detail::top_1_nn_cagra_index index;
  cuvs::cluster::kmeans::detail::build_ann_index<float>(
    f.handle, params, index, centroids_dev, f.ranges.n_clusters, f.ranges.dim);
  const int64_t n = f.ranges.n_rows;
  auto neighbors  = raft::make_device_matrix<uint32_t, int64_t>(f.handle, n, 1);
  auto distances  = raft::make_device_matrix<float, int64_t>(f.handle, n, 1);
  index.search(f.handle, f.dataset.data(), n, neighbors.data_handle(), distances.data_handle());
  std::vector<uint32_t> top1(n);
  raft::copy(top1.data(), neighbors.data_handle(), n, raft::resource::get_cuda_stream(f.handle));
  raft::resource::sync_stream(f.handle);
  return label_match_pct(exact, std::vector<int64_t>(top1.begin(), top1.end()));
}

void BM_LloydFit_Brute(benchmark::State& state)
{
  auto ranges = read_ranges(state);
  FitFixture f(ranges);
  auto params = make_fit_params(ranges.n_clusters, false);
  for (auto _ : state) {
    f.fit(params, f.centroids.data());
  }
  state.SetItemsProcessed(state.iterations() * ranges.n_rows);
  state.counters["inertia_ratio_vs_brute"] = 1.0;
  state.counters["label_match_pct"]        = 100.0;
}

void BM_LloydFit_ANN(benchmark::State& state)
{
  auto ranges      = read_ranges(state);
  const auto itopk = static_cast<size_t>(state.range(3));
  FitFixture f(ranges);
  auto brute_params = make_fit_params(ranges.n_clusters, false);
  auto ann_params   = make_fit_params(ranges.n_clusters, true, itopk);
  rmm::device_uvector<float> brute_centroids(f.centroids.size(),
                                             raft::resource::get_cuda_stream(f.handle));

  double speedup = 0, inertia_ratio = 0, label_match = 0;
  double agree32 = 0, agree64 = 0, agree256 = 0, n_iter_brute = 0, n_iter_ann = 0;

  f.fit(ann_params, f.centroids.data());  // untimed warm-up (first CAGRA use pays one-time costs)

  for (auto _ : state) {
    state.PauseTiming();
    auto [brute_ms, brute_inertia, brute_iters] = f.fit(brute_params, brute_centroids.data());
    auto brute_labels                           = f.predict_labels(brute_centroids.data());
    state.ResumeTiming();

    auto [ann_ms, ann_inertia, ann_iters] = f.fit(ann_params, f.centroids.data());

    state.PauseTiming();
    speedup       = ann_ms > 0 ? brute_ms / ann_ms : 0.0;
    n_iter_brute  = static_cast<double>(brute_iters);
    n_iter_ann    = static_cast<double>(ann_iters);
    inertia_ratio = ann_inertia / brute_inertia;
    label_match   = label_match_pct(brute_labels, f.predict_labels(f.centroids.data()));
    agree32       = ann_1nn_agreement_pct(f, brute_centroids.data(), brute_labels, 32);
    agree64       = ann_1nn_agreement_pct(f, brute_centroids.data(), brute_labels, 64);
    agree256      = ann_1nn_agreement_pct(f, brute_centroids.data(), brute_labels, 256);
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * ranges.n_rows);
  state.counters["speedup_vs_brute"]               = speedup;
  state.counters["n_iter_brute"]                   = n_iter_brute;
  state.counters["n_iter_ann"]                     = n_iter_ann;
  state.counters["inertia_ratio_vs_brute"]         = inertia_ratio;
  state.counters["label_match_pct"]                = label_match;
  state.counters["ann_1nn_agreement_pct_itopk32"]  = agree32;
  state.counters["ann_1nn_agreement_pct_itopk64"]  = agree64;
  state.counters["ann_1nn_agreement_pct_itopk256"] = agree256;
}

// (n_vectors, n_clusters) pairs, all at 5 points per cluster.
constexpr int64_t kSizes[][2] = {{327680, 65536},
                                 {655360, 131072},
                                 {1000000, 200000},
                                 {1500000, 300000},
                                 {2000000, 400000},
                                 {5000000, 1000000}};

constexpr int64_t kItopks[] = {64, 128, 256, 512};

bool register_all()
{
  if (bench_dataset::data_path() == nullptr) {
    std::fprintf(stderr, "Set CUVS_BENCH_DATA_FBIN to the wiki_all .fbin file.\n");
    return false;
  }
  const auto hdr         = bench_dataset::read_header(bench_dataset::data_path());
  const int64_t dim      = hdr.dim;
  const int64_t max_data = 8'000'000'000LL;
  for (auto const& size : kSizes) {
    const int64_t n = size[0];
    const int64_t k = size[1];
    if (n > hdr.n_rows || n * dim * 4 > max_data) { continue; }
    benchmark::RegisterBenchmark("BM_LloydFit_Brute", BM_LloydFit_Brute)
      ->Args({n, k, dim})
      ->ArgNames({"n_vectors", "n_clusters", "dim"})
      ->Unit(benchmark::kMillisecond)
      ->UseRealTime();
    for (int64_t itopk : kItopks) {
      benchmark::RegisterBenchmark("BM_LloydFit_ANN", BM_LloydFit_ANN)
        ->Args({n, k, dim, itopk})
        ->ArgNames({"n_vectors", "n_clusters", "dim", "ann_itopk_size"})
        ->Unit(benchmark::kMillisecond)
        ->UseRealTime();
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv)
{
  if (!register_all()) { return 1; }
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) { return 1; }
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}

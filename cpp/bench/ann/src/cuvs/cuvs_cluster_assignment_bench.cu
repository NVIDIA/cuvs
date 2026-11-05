/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Benchmark: brute force vs CAGRA-based nearest-centroid assignment in Lloyd's k-means
 * (cuvs::cluster::kmeans::predict with kmeans::params::use_ann_for_assignment off / on).
 *
 *  - BM_LloydPredict_BruteForce: kmeans::predict, brute-force top-1 NN.
 *  - BM_LloydPredict_ANN:        kmeans::predict with ANN; pays one index build per call.
 *  - BM_LloydAnn_BuildAndSearch: index build time, search time and label accuracy for each CAGRA
 *    parameter. A Lloyd iteration pays one build plus one search.
 */
#include <benchmark/benchmark.h>

#include "bench_dataset.hpp"

// kmeans_ann_assignment_core.cuh is under cpp/src/; the bench target adds that to the include path
#include <cluster/detail/kmeans_ann_assignment_core.cuh>
#include <cluster/detail/kmeans_common.cuh>
#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <vector>

namespace {

// Vector dim used by every sweep: the dim in the header of CUVS_BENCH_DATA_FBIN.
int64_t g_dim         = 0;
int64_t g_n_rows      = 0;  // rows in CUVS_BENCH_DATA_FBIN
int64_t g_n_centroids = 0;  // centroids in CUVS_BENCH_CENTROIDS_FBIN
// Largest point set a sweep may use; leaves room on a 32 GB GPU for centroids + CAGRA.
constexpr int64_t kMaxDataBytes = 18'000'000'000LL;

bool size_is_supported(int64_t n, int64_t k, int64_t dim)
{
  return dim == g_dim && n <= g_n_rows && k <= g_n_centroids && n * dim * 4 <= kMaxDataBytes;
}

cuvs::cluster::kmeans::params make_params(int64_t n_clusters, bool use_ann)
{
  cuvs::cluster::kmeans::params params;
  params.n_clusters             = static_cast<int>(n_clusters);
  params.metric                 = cuvs::distance::DistanceType::L2Expanded;
  params.use_ann_for_assignment = use_ann;
  return params;
}

struct Fixture {
  raft::device_resources handle;
  rmm::device_uvector<float> X;
  rmm::device_uvector<float> centroids;
  rmm::device_uvector<int64_t> labels;
  int64_t n_rows, n_clusters, dim;

  Fixture(int64_t n, int64_t k, int64_t d)
    : handle{},
      X(static_cast<size_t>(n) * d, raft::resource::get_cuda_stream(handle)),
      centroids(static_cast<size_t>(k) * d, raft::resource::get_cuda_stream(handle)),
      labels(static_cast<size_t>(n), raft::resource::get_cuda_stream(handle)),
      n_rows(n),
      n_clusters(k),
      dim(d)
  {
    bench_dataset::load_data(handle, X.data(), n, d);
    bench_dataset::load_centroids(handle, centroids.data(), k, d);
  }

  void predict(cuvs::cluster::kmeans::params const& params)
  {
    float inertia = 0;
    cuvs::cluster::kmeans::predict(
      handle,
      params,
      raft::make_device_matrix_view<const float, int64_t>(X.data(), n_rows, dim),
      std::nullopt,
      raft::make_device_matrix_view<const float, int64_t>(centroids.data(), n_clusters, dim),
      raft::make_device_vector_view<int64_t, int64_t>(labels.data(), n_rows),
      false,
      raft::make_host_scalar_view(&inertia));
    raft::resource::sync_stream(handle);
  }
};

void run_predict_bench(benchmark::State& state, bool use_ann)
{
  Fixture f(state.range(0), state.range(1), state.range(2));
  auto params = make_params(f.n_clusters, use_ann);
  f.predict(params);  // untimed warm-up (one-time kernel/allocator costs)
  for (auto _ : state) {
    f.predict(params);
  }
  state.SetItemsProcessed(state.iterations() * f.n_rows);
}

void BM_LloydPredict_BruteForce(benchmark::State& state) { run_predict_bench(state, false); }
void BM_LloydPredict_ANN(benchmark::State& state) { run_predict_bench(state, true); }

// Times the two parts of one ANN assignment (what a Lloyd iteration pays for it), and reports the
// accuracy:
//  - build_ms:  building the CAGRA index over the centroids,
//  - search_ms: the top-1 search of all points (best of 3),
//  - match_pct: % of points whose ANN label equals the brute-force label.
// Args: n, k, dim, graph_degree, itopk.
void BM_LloydAnn_BuildAndSearch(benchmark::State& state)
{
  Fixture f(state.range(0), state.range(1), state.range(2));
  auto params                          = make_params(f.n_clusters, true);
  params.ann_graph_degree              = static_cast<size_t>(state.range(3));
  params.ann_intermediate_graph_degree = 2 * params.ann_graph_degree;
  params.ann_itopk_size                = static_cast<size_t>(state.range(4));
  auto centers_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    f.centroids.data(), f.n_clusters, f.dim);
  auto X_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    f.X.data(), f.n_rows, f.dim);
  auto stream = raft::resource::get_cuda_stream(f.handle);
  rmm::device_uvector<int64_t> labels(f.n_rows, stream);
  rmm::device_uvector<float> dists(f.n_rows, stream);
  auto timed = [&](auto&& fn) {
    raft::resource::sync_stream(f.handle);
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    raft::resource::sync_stream(f.handle);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  };
  auto to_host = [&](int64_t* dev) {
    std::vector<int64_t> h(f.n_rows);
    raft::copy(h.data(), dev, f.n_rows, stream);
    raft::resource::sync_stream(f.handle);
    return h;
  };

  f.predict(make_params(f.n_clusters, false));
  const auto brute_labels = to_host(f.labels.data());

  for (auto _ : state) {
    {
      cuvs::distance::detail::top_1_nn_cagra_index warm;
      cuvs::cluster::kmeans::detail::build_ann_index<float>(
        f.handle, params, warm, f.centroids.data(), f.n_clusters, f.dim);  // untimed warm-up

      raft::resource::sync_stream(f.handle);
    }
    cuvs::distance::detail::top_1_nn_cagra_index index;
    const double build_ms = timed([&]() {
      cuvs::cluster::kmeans::detail::build_ann_index<float>(
        f.handle, params, index, f.centroids.data(), f.n_clusters, f.dim);
    });
    double search_ms      = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
      search_ms = std::min(
        search_ms, timed([&]() {
          // The ANN assignment is the Cagra backend of top_1_nn, reached through the
          // same minClusterAndDistanceCompute the brute-force E-step uses.
          rmm::device_uvector<char> out_storage(0, stream), ws(0, stream);
          rmm::device_uvector<float> norm_buf(0, stream);
          auto x_norm = raft::make_device_vector_view<const float, int64_t>(
            static_cast<const float*>(nullptr), f.n_rows);
          const auto result =
            cuvs::cluster::kmeans::detail::minClusterAndDistanceCompute<float, int64_t>(
              f.handle,
              X_view,
              centers_view,
              out_storage,
              x_norm,
              norm_buf,
              params.metric,
              params.batch_samples,
              params.batch_centroids,
              ws,
              std::nullopt,
              cuvs::distance::detail::Top1nnBackend::Cagra,
              &index);
          cuvs::cluster::kmeans::detail::copyClusterLabels(f.handle, result, labels.data());
        }));
    }
    const auto ann_labels = to_host(labels.data());
    int64_t same          = 0;
    for (int64_t i = 0; i < f.n_rows; ++i) {
      same += (ann_labels[i] == brute_labels[i]);
    }
    state.SetIterationTime((build_ms + search_ms) / 1e3);
    state.counters["build_ms"]  = build_ms;
    state.counters["search_ms"] = search_ms;
    state.counters["match_pct"] = 100.0 * static_cast<double>(same) / static_cast<double>(f.n_rows);
  }
}

// Controlled sweeps. Points-per-cluster is held fixed in the first so K is the only variable;
// the second holds N fixed and varies K alone.
constexpr int64_t kPointsPerCluster = 5;
constexpr int64_t kFixedN           = 2000000;
// clang-format off
constexpr int64_t kClusterCounts[]       = {1000, 4000, 16000, 65536, 131072, 262144, 500000, 1000000};
// clang-format on

void register_pair(int64_t n, int64_t k)
{
  for (auto const& [name, fn] : std::vector<std::pair<const char*, void (*)(benchmark::State&)>>{
         {"BM_LloydPredict_BruteForce", BM_LloydPredict_BruteForce},
         {"BM_LloydPredict_ANN", BM_LloydPredict_ANN}}) {
    benchmark::RegisterBenchmark(name, fn)
      ->Args({n, k, g_dim})
      ->ArgNames({"n_vectors", "n_clusters", "dim"})
      ->Unit(benchmark::kMillisecond)
      ->UseRealTime();
  }
}

void register_sweeps()
{
  for (int64_t k : kClusterCounts) {
    int64_t n = kPointsPerCluster * k;
    if (size_is_supported(n, k, g_dim)) { register_pair(n, k); }
    if (size_is_supported(kFixedN, k, g_dim)) { register_pair(kFixedN, k); }
  }
  // Build/search time and accuracy at the default settings, then one parameter varied at a time.
  // clang-format off
  constexpr int64_t kGraphDegrees[] = {32, 64, 128};
  constexpr int64_t kItopks[]       = {32, 64, 128, 256};
  // clang-format on
  auto add = [](int64_t k, int64_t gd, int64_t itopk) {
    benchmark::RegisterBenchmark("BM_LloydAnn_BuildAndSearch", BM_LloydAnn_BuildAndSearch)
      ->Args({kFixedN, k, g_dim, gd, itopk})
      ->ArgNames({"n_vectors", "n_clusters", "dim", "graph_degree", "itopk"})
      ->Unit(benchmark::kMillisecond)
      ->UseManualTime()
      ->Iterations(1);
  };
  for (int64_t k : kClusterCounts) {
    if (!size_is_supported(kFixedN, k, g_dim)) { continue; }
    add(k, 64, 64);
    if (k != 65536 && k != 262144) { continue; }  // vary parameters at two representative K
    for (int64_t gd : kGraphDegrees) {
      if (gd != 64) { add(k, gd, 64); }
    }
    for (int64_t it : kItopks) {
      if (it != 64) { add(k, 64, it); }
    }
  }
}

// Trains centroids with brute-force Lloyd's k-means on the first CUVS_BENCH_TRAIN_N rows of
// CUVS_BENCH_DATA_FBIN and writes them to CUVS_BENCH_TRAIN_CENTROIDS_OUT as .fbin, so the
// benchmarks can run on centroids with the dataset's structure. CUVS_BENCH_TRAIN_N defaults to
// 5M and CUVS_BENCH_TRAIN_K to 300K.
int train_centroids_from_env()
{
  const char* out_path = std::getenv("CUVS_BENCH_TRAIN_CENTROIDS_OUT");
  if (bench_dataset::data_path() == nullptr) {
    std::fprintf(stderr, "CUVS_BENCH_TRAIN_CENTROIDS_OUT also needs CUVS_BENCH_DATA_FBIN\n");
    return 1;
  }
  const char* n_env     = std::getenv("CUVS_BENCH_TRAIN_N");
  const char* k_env     = std::getenv("CUVS_BENCH_TRAIN_K");
  const int64_t n_train = n_env ? std::atoll(n_env) : 5000000;
  const int64_t k       = k_env ? std::atoll(k_env) : 300000;

  const auto hdr = bench_dataset::read_header(bench_dataset::data_path());
  raft::device_resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);
  rmm::device_uvector<float> X(static_cast<size_t>(n_train) * hdr.dim, stream);
  rmm::device_uvector<float> centroids(static_cast<size_t>(k) * hdr.dim, stream);
  bench_dataset::load_data(handle, X.data(), n_train, hdr.dim);

  auto params     = make_params(k, false);
  params.max_iter = 10;
  params.init     = cuvs::cluster::kmeans::params::Random;
  float inertia   = 0;
  int64_t n_iter  = 0;
  cuvs::cluster::kmeans::fit(
    handle,
    params,
    raft::make_device_matrix_view<const float, int64_t>(X.data(), n_train, hdr.dim),
    std::nullopt,
    raft::make_device_matrix_view<float, int64_t>(centroids.data(), k, hdr.dim),
    raft::make_host_scalar_view(&inertia),
    raft::make_host_scalar_view(&n_iter));
  raft::resource::sync_stream(handle);

  std::vector<float> host(static_cast<size_t>(k) * hdr.dim);
  raft::copy(host.data(), centroids.data(), host.size(), stream);
  raft::resource::sync_stream(handle);
  bench_dataset::write_fbin(out_path, host.data(), k, hdr.dim);
  std::printf("trained %lld centroids (dim %lld) on %lld points -> %s\n",
              static_cast<long long>(k),
              static_cast<long long>(hdr.dim),
              static_cast<long long>(n_train),
              out_path);
  return 0;
}

bool configure_dataset_from_env()
{
  if (bench_dataset::data_path() == nullptr || bench_dataset::centroids_path() == nullptr) {
    std::fprintf(
      stderr,
      "Set CUVS_BENCH_DATA_FBIN to the wiki_all .fbin file and CUVS_BENCH_CENTROIDS_FBIN to "
      "centroids trained on it (run this binary with CUVS_BENCH_TRAIN_CENTROIDS_OUT "
      "set to create them).\n");
    return false;
  }
  const auto data      = bench_dataset::read_header(bench_dataset::data_path());
  const auto centroids = bench_dataset::read_header(bench_dataset::centroids_path());
  g_n_rows             = data.n_rows;
  g_n_centroids        = centroids.n_rows;
  g_dim                = data.dim;
  return true;
}

}  // namespace

int main(int argc, char** argv)
{
  if (std::getenv("CUVS_BENCH_TRAIN_CENTROIDS_OUT") != nullptr) {
    return train_centroids_from_env();
  }
  if (!configure_dataset_from_env()) { return 1; }
  register_sweeps();
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) { return 1; }
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}

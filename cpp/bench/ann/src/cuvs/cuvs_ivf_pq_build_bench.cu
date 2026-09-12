/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * IVF-PQ nearest-centroid lookup benchmarks (three scenarios):
 *
 * All compare brute force vs CAGRA for the same primitive: given centroid vectors, find the
 * nearest centroid for each data vector. CAGRA never computes centroids; it only accelerates
 * lookup.
 *
 *  - Build fit: lookup during k-means fit E-step (`use_ann_for_build_fit`). Centroids move each
 *    EM iteration; CAGRA index is rebuilt periodically. Times full `build()`.
 *  - Build post-fit: lookup after fit to label the train subsample for PQ codebooks
 *    (`use_ann_for_build_postfit`). Fixed centroids; CAGRA built once per build. Times full
 *    `build()`.
 *  - Extend: fixed trained centroids (`use_ann_for_extend`). CAGRA built once per extend batch
 *    loop. Times `extend()` only; setup deserialize is not timed.
 */
#include <benchmark/benchmark.h>

#include <cuvs/neighbors/ivf_pq.hpp>

#include <chrono>
#include <functional>
#include <memory>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/util/cudart_utils.hpp>
#include <sstream>
#include <string>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_uvector.hpp>

namespace {

struct BenchRanges {
  int64_t n_rows;
  uint32_t n_lists;
  int64_t dim;
};

BenchRanges read_ranges(benchmark::State const& state)
{
  return BenchRanges{static_cast<int64_t>(state.range(0)),
                     static_cast<uint32_t>(state.range(1)),
                     static_cast<int64_t>(state.range(2))};
}

void init_random_dataset(raft::resources const& handle, float* data, int64_t n_rows, int64_t dim)
{
  raft::random::RngState rng(12345ULL);
  raft::random::uniform(handle, rng, data, n_rows * dim, float(-1), float(1));
  raft::resource::sync_stream(handle);
}

/** Shared dataset + handle setup for both benchmarks. */
struct DatasetFixture {
  raft::device_resources handle;
  rmm::device_uvector<float> dataset;
  raft::device_matrix_view<const float, int64_t, raft::row_major> view;

  explicit DatasetFixture(BenchRanges const& ranges)
    : handle{},
      dataset(static_cast<size_t>(ranges.n_rows) * static_cast<size_t>(ranges.dim),
              raft::resource::get_cuda_stream(handle))
  {
    init_random_dataset(handle, dataset.data(), ranges.n_rows, ranges.dim);
    raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
    view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
      dataset.data(), ranges.n_rows, ranges.dim);
  }
};

void set_common_index_params(cuvs::neighbors::ivf_pq::index_params& params, uint32_t n_lists)
{
  params.n_lists                  = n_lists;
  params.kmeans_n_iters           = 3;
  params.kmeans_trainset_fraction = 0.2;
  params.metric                   = cuvs::distance::DistanceType::L2Expanded;
}

/** build() fit E-step only: toggles use_ann_for_build_fit; post-fit and extend stay brute. */
cuvs::neighbors::ivf_pq::index_params make_build_fit_lookup_params(uint32_t n_lists,
                                                                   bool use_ann_for_build_fit)
{
  cuvs::neighbors::ivf_pq::index_params params;
  set_common_index_params(params, n_lists);
  params.add_data_on_build         = true;
  params.use_ann_for_build_fit     = use_ann_for_build_fit;
  params.use_ann_for_build_postfit = false;
  params.use_ann_for_extend        = false;
  return params;
}

/** build() post-fit predict only: toggles use_ann_for_build_postfit; fit E-step and extend stay
 * brute. */
cuvs::neighbors::ivf_pq::index_params make_build_postfit_lookup_params(
  uint32_t n_lists, bool use_ann_for_build_postfit)
{
  cuvs::neighbors::ivf_pq::index_params params;
  set_common_index_params(params, n_lists);
  params.add_data_on_build         = true;
  params.use_ann_for_build_fit     = false;
  params.use_ann_for_build_postfit = use_ann_for_build_postfit;
  params.use_ann_for_extend        = false;
  return params;
}

cuvs::neighbors::ivf_pq::index_params make_extend_lookup_params(uint32_t n_lists,
                                                                bool use_ann_for_extend)
{
  cuvs::neighbors::ivf_pq::index_params params;
  set_common_index_params(params, n_lists);
  params.add_data_on_build         = false;
  params.use_ann_for_build_fit     = false;
  params.use_ann_for_build_postfit = false;
  params.use_ann_for_extend        = use_ann_for_extend;
  return params;
}

std::string serialize_index_blob(raft::resources const& handle,
                                 cuvs::neighbors::ivf_pq::index<int64_t> const& index)
{
  std::ostringstream os(std::ios::binary);
  cuvs::neighbors::ivf_pq::serialize(handle, os, index);
  raft::resource::sync_stream(handle);
  return os.str();
}

void deserialize_index_from_blob(raft::resources const& handle,
                                 std::string const& blob,
                                 cuvs::neighbors::ivf_pq::index<int64_t>* index)
{
  std::istringstream is(blob, std::ios::binary);
  cuvs::neighbors::ivf_pq::deserialize(handle, is, index);
  raft::resource::sync_stream(handle);
}

/** Serialized empty trained indices for per-iteration extend reset (setup, not timed). */
struct ExtendIndexSnapshots {
  std::string empty_index_snapshot_bf;
  std::string empty_index_snapshot_cagra;

  static ExtendIndexSnapshots create(
    raft::resources const& handle,
    cuvs::neighbors::ivf_pq::index_params const& params_bf,
    cuvs::neighbors::ivf_pq::index_params const& params_cagra,
    raft::device_matrix_view<const float, int64_t, raft::row_major> dataset_view)
  {
    ExtendIndexSnapshots snapshots;
    auto idx_bf    = cuvs::neighbors::ivf_pq::build(handle, params_bf, dataset_view);
    auto idx_cagra = cuvs::neighbors::ivf_pq::build(handle, params_cagra, dataset_view);
    snapshots.empty_index_snapshot_bf    = serialize_index_blob(handle, idx_bf);
    snapshots.empty_index_snapshot_cagra = serialize_index_blob(handle, idx_cagra);
    return snapshots;
  }

  void restore(raft::resources const& handle,
               cuvs::neighbors::ivf_pq::index<int64_t>* index_bf,
               cuvs::neighbors::ivf_pq::index<int64_t>* index_cagra) const
  {
    deserialize_index_from_blob(handle, empty_index_snapshot_bf, index_bf);
    deserialize_index_from_blob(handle, empty_index_snapshot_cagra, index_cagra);
  }
};

double time_synced_ms(raft::resources const& handle, std::function<void()> const& op)
{
  auto start = std::chrono::steady_clock::now();
  op();
  raft::resource::sync_stream(handle);
  auto end = std::chrono::steady_clock::now();
  return 1e-6 * std::chrono::duration<double, std::nano>(end - start).count();
}

/** Run brute vs CAGRA paths each iteration; `before_timed_work` runs outside the chrono window. */
template <typename PreIterationFn, typename BfFn, typename CagraFn>
void accumulate_bf_vs_cagra_ms(benchmark::State& state,
                               raft::resources const& handle,
                               PreIterationFn&& before_timed_work,
                               BfFn&& run_bf,
                               CagraFn&& run_cagra,
                               double& total_bf_ms,
                               double& total_cagra_ms)
{
  for (auto _ : state) {
    before_timed_work(state);
    total_bf_ms += time_synced_ms(handle, run_bf);
    total_cagra_ms += time_synced_ms(handle, run_cagra);
  }
}

void set_speedup_counters(benchmark::State& state,
                          char const* speedup_key,
                          char const* bf_key,
                          char const* cagra_key,
                          double total_bf_ms,
                          double total_cagra_ms)
{
  if (total_cagra_ms > 0) { state.counters[speedup_key] = total_bf_ms / total_cagra_ms; }
  state.counters[bf_key] = benchmark::Counter(total_bf_ms, benchmark::Counter::kAvgIterations);
  state.counters[cagra_key] =
    benchmark::Counter(total_cagra_ms, benchmark::Counter::kAvgIterations);
}

void run_build_bf_vs_cagra_speedup(benchmark::State& state,
                                   DatasetFixture const& fixture,
                                   cuvs::neighbors::ivf_pq::index_params const& params_bf,
                                   cuvs::neighbors::ivf_pq::index_params const& params_cagra,
                                   char const* speedup_key,
                                   char const* bf_key,
                                   char const* cagra_key)
{
  double total_bf_ms = 0.0, total_cagra_ms = 0.0;
  accumulate_bf_vs_cagra_ms(
    state,
    fixture.handle,
    [](benchmark::State&) {},
    [&] {
      auto idx = cuvs::neighbors::ivf_pq::build(fixture.handle, params_bf, fixture.view);
      benchmark::DoNotOptimize(idx.size());
    },
    [&] {
      auto idx = cuvs::neighbors::ivf_pq::build(fixture.handle, params_cagra, fixture.view);
      benchmark::DoNotOptimize(idx.size());
    },
    total_bf_ms,
    total_cagra_ms);
  set_speedup_counters(state, speedup_key, bf_key, cagra_key, total_bf_ms, total_cagra_ms);
}

}  // namespace

/**
 * Full IVF-PQ build: brute vs CAGRA nearest-centroid lookup during k-means fit E-step only
 * (`use_ann_for_build_fit`). Post-fit predict and add_data_on_build stay brute. Centroids move
 * each fit iteration, so CAGRA is rebuilt across EM iterations. Times all of `build()`; PQ
 * training and other steps dilute the measured speedup.
 */
static void BM_IVFPQ_BuildFit_NearestCentroidLookup_Speedup(benchmark::State& state)
{
  auto ranges = read_ranges(state);
  DatasetFixture fixture(ranges);
  run_build_bf_vs_cagra_speedup(state,
                                fixture,
                                make_build_fit_lookup_params(ranges.n_lists, false),
                                make_build_fit_lookup_params(ranges.n_lists, true),
                                "speedup_build_fit",
                                "bf_build_fit_ms",
                                "cagra_build_fit_ms");
}

/**
 * Full IVF-PQ build: brute vs CAGRA nearest-centroid lookup during post-fit predict only
 * (`use_ann_for_build_postfit`). Fit E-step and add_data_on_build stay brute. Centroids are
 * fixed; CAGRA is built once per build for train-subsample labeling. Times all of `build()`;
 * k-means fit, PQ training, and other steps dilute the measured speedup.
 */
static void BM_IVFPQ_BuildPostfit_NearestCentroidLookup_Speedup(benchmark::State& state)
{
  auto ranges = read_ranges(state);
  DatasetFixture fixture(ranges);
  run_build_bf_vs_cagra_speedup(state,
                                fixture,
                                make_build_postfit_lookup_params(ranges.n_lists, false),
                                make_build_postfit_lookup_params(ranges.n_lists, true),
                                "speedup_build_postfit",
                                "bf_build_postfit_ms",
                                "cagra_build_postfit_ms");
}

/**
 * extend() only: brute vs CAGRA nearest-centroid lookup with fixed trained centroids
 * (`use_ann_for_extend`). CAGRA is built once; new vectors are assigned via fast 1-NN search.
 * Empty trained indices are restored each iteration via deserialize (not timed).
 */
static void BM_IVFPQ_Extend_NearestCentroidLookup_Speedup(benchmark::State& state)
{
  auto ranges = read_ranges(state);
  DatasetFixture fixture(ranges);
  auto params_bf    = make_extend_lookup_params(ranges.n_lists, false);
  auto params_cagra = make_extend_lookup_params(ranges.n_lists, true);
  auto snapshots =
    ExtendIndexSnapshots::create(fixture.handle, params_bf, params_cagra, fixture.view);

  cuvs::neighbors::ivf_pq::index<int64_t> index_bf(fixture.handle);
  cuvs::neighbors::ivf_pq::index<int64_t> index_cagra(fixture.handle);

  double total_bf_ms = 0.0, total_cagra_ms = 0.0;
  accumulate_bf_vs_cagra_ms(
    state,
    fixture.handle,
    [&](benchmark::State& st) {
      st.PauseTiming();
      snapshots.restore(fixture.handle, &index_bf, &index_cagra);
      st.ResumeTiming();
    },
    [&] { cuvs::neighbors::ivf_pq::extend(fixture.handle, fixture.view, std::nullopt, &index_bf); },
    [&] {
      cuvs::neighbors::ivf_pq::extend(fixture.handle, fixture.view, std::nullopt, &index_cagra);
    },
    total_bf_ms,
    total_cagra_ms);
  set_speedup_counters(
    state, "speedup_extend", "bf_extend_ms", "cagra_extend_ms", total_bf_ms, total_cagra_ms);
}

constexpr int64_t kDim = 128;

#define IVFPQ_LOOKUP_BENCH_ARGS(BM)              \
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

// build() fit E-step only (full build timed).
IVFPQ_LOOKUP_BENCH_ARGS(BM_IVFPQ_BuildFit_NearestCentroidLookup_Speedup)

// build() post-fit predict only (full build timed).
IVFPQ_LOOKUP_BENCH_ARGS(BM_IVFPQ_BuildPostfit_NearestCentroidLookup_Speedup)

// extend(): fixed-centroid nearest-centroid lookup, CAGRA built once per extend.
IVFPQ_LOOKUP_BENCH_ARGS(BM_IVFPQ_Extend_NearestCentroidLookup_Speedup)

BENCHMARK_MAIN();

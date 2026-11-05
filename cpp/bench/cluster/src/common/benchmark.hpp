/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../../../ann/src/common/dataset.hpp"
#include "../../../ann/src/common/util.hpp"
#include "../../../common/benchmark_utils.hpp"
#include "conf.hpp"

#include <benchmark/benchmark.h>
#include <cuda_fp16.h>
#include <cuvs/cluster/kmeans.hpp>
#include <nlohmann/json.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/sample_rows.cuh>
#include <raft/random/rng_state.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace cuvs::bench::cluster {

/** Seed of the `ratio` subsampling, fixed so that every case of a run fits the same rows. */
constexpr uint64_t kSampleSeed = 42;

/** A clustering algorithm under benchmark; `fit` is the timed region. */
template <typename T>
class algo {
 public:
  virtual ~algo() noexcept = default;

  /** Fit on device-resident data; must return only after the GPU work has completed. */
  virtual void fit(raft::device_matrix_view<const T, int64_t> X) = 0;

  /** Centroids found by the last `fit`. */
  [[nodiscard]] virtual auto centroids() const
    -> raft::device_matrix_view<const float, int64_t> = 0;

  /** Convergence numbers of the last `fit` (e.g. iterations), reported as counters. */
  [[nodiscard]] virtual auto metrics() const -> std::vector<std::tuple<std::string, double>> = 0;
};

/** Implemented by each benchmark executable. Throws if `name` or `T` is not supported. */
template <typename T>
auto create_algo(const std::string& name, const nlohmann::json& param, const raft::resources& res)
  -> std::unique_ptr<algo<T>>;

template <typename T>
void bench_fit(::benchmark::State& state,
               std::shared_ptr<const bench::dataset<T>> data,
               configuration::clustering conf,
               std::shared_ptr<raft::device_resources> res)
{
  dump_parameters(state, conf.param);

  std::unique_ptr<algo<T>> a;
  try {
    a = create_algo<T>(conf.algo, conf.param, *res);
  } catch (const std::exception& e) {
    return state.SkipWithError("Failed to create an algo: " + std::string(e.what()));
  }

  auto X = raft::make_device_matrix_view<const T, int64_t>(
    data->base_set(MemoryType::kDevice), data->base_set_size(), data->dim());

  // `ratio` fits on a random 1/ratio of the rows (as the IVF builds train their k-means), drawn
  // once per case and outside the timed loop. Like the IVF builds, keep at least n_clusters rows.
  auto X_fit = X;
  std::optional<raft::device_matrix<T, int64_t>> trainset;
  try {
    const double ratio = conf.param.value("ratio", 1.0);
    if (ratio < 1.0) { throw std::invalid_argument("ratio must be >= 1"); }
    const auto n_train = std::max<int64_t>(static_cast<int64_t>(X.extent(0) / ratio),
                                           conf.param.value("n_clusters", int64_t{0}));
    if (n_train < X.extent(0)) {
      trainset.emplace(raft::make_device_matrix<T, int64_t>(*res, n_train, X.extent(1)));
      raft::matrix::sample_rows<T, int64_t>(
        *res, raft::random::RngState{kSampleSeed}, X, trainset->view());
      X_fit = raft::make_const_mdspan(trainset->view());
    }
  } catch (const std::exception& e) {
    return state.SkipWithError("Failed to subsample: " + std::string(e.what()));
  }

  for (auto _ : state) {
    raft::common::nvtx::range<raft::common::nvtx::domain::app> lap("%s", state.name().c_str());
    try {
      a->fit(X_fit);
      raft::resource::sync_stream(*res);
    } catch (const std::exception& e) {
      state.SkipWithError(std::string(e.what()));
      break;
    }
  }
  if (state.skipped()) { return; }

  for (auto& [key, val] : a->metrics()) {
    state.counters.insert({{key, val}});
  }
  // Inertia is a quality measure, not part of the work being timed: compute it after the timed
  // loop, over the full dataset so that subsampled fits are judged on the data they summarize.
  if constexpr (std::is_same_v<T, float>) {
    float inertia = 0;
    cuvs::cluster::kmeans::cluster_cost(
      *res, X, a->centroids(), raft::make_host_scalar_view(&inertia));
    state.counters.insert({{"inertia", inertia}});
  }
  state.counters.insert({{"n_rows", static_cast<double>(X_fit.extent(0))},
                         {"dim", static_cast<double>(X_fit.extent(1))}});
}

template <typename T>
void dispatch_benchmark(const std::string& cmdline,
                        configuration& conf,
                        const kv_series& override_kv,
                        std::shared_ptr<raft::device_resources> res)
{
  ::benchmark::AddCustomContext("command_line", cmdline);
  for (auto [key, value] : host_info()) {
    ::benchmark::AddCustomContext(key, value);
  }
  for (auto [key, value] : cuda_info()) {
    ::benchmark::AddCustomContext(key, value);
  }

  const auto& dc = conf.get_dataset_conf();
  // No query set: only the base set is read, and only when first accessed.
  auto data = std::make_shared<const bench::dataset<T>>(
    dc.name, dc.base_file, dc.subset_first_row, dc.subset_size, "", "euclidean", std::nullopt);
  log_info("Using the dataset file '%s'", dc.base_file.c_str());
  // Load to the device here so that no benchmark case times the file read.
  [[maybe_unused]] auto* base_set = data->base_set(MemoryType::kDevice);
  ::benchmark::AddCustomContext("dataset", dc.name);
  ::benchmark::AddCustomContext("dtype", dc.dtype);
  ::benchmark::AddCustomContext("n_records", std::to_string(data->base_set_size()));
  ::benchmark::AddCustomContext("dim", std::to_string(data->dim()));

  for (const auto& c : conf.get_clusterings()) {
    for (auto param : apply_overrides(std::vector{c.param}, override_kv)) {
      auto suffix = static_cast<std::string>(param["override_suffix"]);
      param.erase("override_suffix");
      auto modified  = c;
      modified.param = param;
      ::benchmark::RegisterBenchmark(c.name + suffix, bench_fit<T>, data, modified, res)
        ->Unit(::benchmark::kMillisecond)
        // fit() blocks until the GPU is done, so the wall clock is the end-to-end fit time.
        ->UseRealTime()
        ->MeasureProcessCPUTime();
    }
  }
}

inline void printf_usage()
{
  ::benchmark::PrintDefaultHelp();
  std::fprintf(
    stdout,
    "          [--data_prefix=<prefix>]\n"
    "          [--override_kv=<key:value1:value2:...:valueN>]\n"
    "          [--no_pool]\n"
    "          <conf>.json\n"
    "\n"
    "Note the non-standard benchmark parameters:\n"
    "  --data_prefix=<prefix>: prepend <prefix> to dataset file paths specified in the <conf>.json"
    " (default = 'data/').\n"
    "  --override_kv=<key:value1:value2:...:valueN>: override a clustering param one or more"
    " times multiplying the number of configurations; you can use this parameter multiple times"
    " to get the Cartesian product of benchmark configs.\n"
    "  --no_pool: allocate with the default device memory resource instead of an RMM pool.\n");
}

inline auto run_main(int argc, char** argv) -> int
{
  std::string data_prefix     = "data";
  std::string new_override_kv = "";
  bool no_pool                = false;
  kv_series override_kv{};

  if (argc < 2) {
    printf_usage();
    return -1;
  }
  // Save command line for reproducibility.
  std::string cmdline(argv[0]);
  for (int i = 1; i < argc; i++) {
    cmdline += " " + std::string(argv[i]);
  }

  char* conf_path = argv[--argc];
  std::ifstream conf_stream(conf_path);

  for (int i = 1; i < argc; i++) {
    if (parse_bool_flag(argv[i], "--no_pool", no_pool) ||
        parse_string_flag(argv[i], "--data_prefix", data_prefix) ||
        parse_string_flag(argv[i], "--override_kv", new_override_kv)) {
      if (!new_override_kv.empty()) {
        auto kvv = split(new_override_kv, ':');
        if (kvv.size() < 2) {
          log_error("Invalid --override_kv=%s, expected <key>:<value1>[:<value2>...]",
                    new_override_kv.c_str());
          return -1;
        }
        std::vector<nlohmann::json> vals{};
        for (std::size_t j = 1; j < kvv.size(); j++) {
          try {
            vals.push_back(nlohmann::json::parse(kvv[j]));
          } catch (const nlohmann::json::parse_error&) {
            log_error("Invalid --override_kv value '%s', expected JSON (quote strings: '\"%s\"')",
                      kvv[j].c_str(),
                      kvv[j].c_str());
            return -1;
          }
        }
        override_kv.emplace_back(kvv[0], vals);
        new_override_kv = "";
      }
      for (int j = i; j < argc - 1; j++) {
        argv[j] = argv[j + 1];
      }
      argc--;
      i--;
    }
  }

  if (!conf_stream) {
    log_error("Can't open configuration file: %s", conf_path);
    return -1;
  }

  // The pool keeps cudaMalloc/cudaFree of the per-fit workspaces out of the measured time.
  std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> orig_mr;
  if (!no_pool) {
    orig_mr = rmm::mr::set_current_device_resource(rmm::mr::pool_memory_resource{
      rmm::mr::get_current_device_resource_ref(), std::size_t{1} << 30});
  }

  int ret = 0;
  {
    auto res = std::make_shared<raft::device_resources>();
    try {
      configuration conf(conf_stream, data_prefix);
      const auto& dtype = conf.get_dataset_conf().dtype;
      if (dtype == "float") {
        dispatch_benchmark<float>(cmdline, conf, override_kv, res);
      } else if (dtype == "half") {
        dispatch_benchmark<half>(cmdline, conf, override_kv, res);
      } else if (dtype == "int8") {
        dispatch_benchmark<std::int8_t>(cmdline, conf, override_kv, res);
      } else if (dtype == "uint8") {
        dispatch_benchmark<std::uint8_t>(cmdline, conf, override_kv, res);
      } else {
        throw std::runtime_error("datatype '" + dtype + "' is not supported");
      }
      ::benchmark::Initialize(&argc, argv, printf_usage);
      if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        ret = -1;
      } else {
        ::benchmark::RunSpecifiedBenchmarks();
      }
      ::benchmark::Shutdown();
    } catch (const std::exception& e) {
      log_error("%s", e.what());
      ret = -1;
    }
    // Registered benchmarks hold the dataset and resources; release them before the pool.
    ::benchmark::ClearRegisteredBenchmarks();
  }
  if (orig_mr.has_value()) { rmm::mr::set_current_device_resource(std::move(*orig_mr)); }
  return ret;
}

}  // namespace cuvs::bench::cluster

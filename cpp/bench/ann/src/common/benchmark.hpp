/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "ann_types.hpp"
#include "conf.hpp"
#include "dataset.hpp"
#include "nvtx_stats.hpp"
#include "util.hpp"

#include <benchmark/benchmark.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace cuvs::bench {

static inline std::unique_ptr<algo_base> current_algo{nullptr};
static inline std::unique_ptr<algo_property> current_algo_props{nullptr};

using kv_series = std::vector<std::tuple<std::string, std::vector<nlohmann::json>>>;

inline auto apply_overrides(const std::vector<nlohmann::json>& configs,
                            const kv_series& overrides,
                            std::size_t override_idx = 0) -> std::vector<nlohmann::json>
{
  std::vector<nlohmann::json> results{};
  if (override_idx >= overrides.size()) {
    auto n = configs.size();
    for (size_t i = 0; i < n; i++) {
      auto c               = configs[i];
      c["override_suffix"] = n > 1 ? "/" + std::to_string(i) : "";
      results.push_back(c);
    }
    return results;
  }
  auto rec_configs = apply_overrides(configs, overrides, override_idx + 1);
  auto [key, vals] = overrides[override_idx];
  auto n           = vals.size();
  for (size_t i = 0; i < n; i++) {
    const auto& val = vals[i];
    for (auto rc : rec_configs) {
      if (n > 1) {
        rc["override_suffix"] =
          static_cast<std::string>(rc["override_suffix"]) + "/" + std::to_string(i);
      }
      rc[key] = val;
      results.push_back(rc);
    }
  }
  return results;
}

inline auto apply_overrides(const nlohmann::json& config,
                            const kv_series& overrides,
                            std::size_t override_idx = 0)
{
  return apply_overrides(std::vector{config}, overrides, 0);
}

inline void dump_parameters(::benchmark::State& state, nlohmann::json params)
{
  std::string label = "";
  bool label_empty  = true;
  for (auto& [key, val] : params.items()) {
    if (val.is_number()) {
      state.counters.insert({{key, val}});
    } else if (val.is_boolean()) {
      state.counters.insert({{key, val ? 1.0 : 0.0}});
    } else {
      auto kv = key + "=" + val.dump();
      if (label_empty) {
        label = kv;
      } else {
        label += "#" + kv;
      }
      label_empty = false;
    }
  }
  if (!label_empty) { state.SetLabel(label); }
}

inline auto parse_algo_property(algo_property prop, const nlohmann::json& conf) -> algo_property
{
  if (conf.contains("dataset_memory_type")) {
    prop.dataset_memory_type = parse_memory_type(conf.at("dataset_memory_type"));
  }
  if (conf.contains("query_memory_type")) {
    prop.query_memory_type = parse_memory_type(conf.at("query_memory_type"));
  }
  return prop;
};

template <typename T>
void bench_build(::benchmark::State& state,
                 std::shared_ptr<const dataset<T>> dataset,
                 const configuration::index& index,
                 bool force_overwrite,
                 bool no_lap_sync)
{
  // NB: these two thread-local vars can be used within algo wrappers
  cuvs::bench::benchmark_thread_id = state.thread_index();
  cuvs::bench::benchmark_n_threads = state.threads();
  dump_parameters(state, index.build_param);

  if (state.thread_index() == 0) {
    log_info(
      "bench_build start: algo=%s index=%s file=%s force_overwrite=%d threads=%d dim=%d "
      "base_set_size=%zu",
      index.algo.c_str(),
      index.name.c_str(),
      index.file.c_str(),
      force_overwrite ? 1 : 0,
      state.threads(),
      dataset->dim(),
      dataset->base_set_size());
  }

  if (file_exists(index.file)) {
    if (force_overwrite) {
      log_info("bench_build overwriting existing index file: %s", index.file.c_str());
    } else {
      log_info("bench_build skipped existing index file: %s", index.file.c_str());
      return state.SkipWithMessage(
        "Index file already exists (use --force to overwrite the index).");
    }
  }

  std::unique_ptr<algo<T>> algo;
  try {
    if (state.thread_index() == 0) {
      log_info(
        "bench_build create algo start: algo=%s index=%s", index.algo.c_str(), index.name.c_str());
    }
    algo = create_algo<T>(index.algo, dataset->distance(), dataset->dim(), index.build_param);
    if (state.thread_index() == 0) {
      log_info(
        "bench_build create algo finish: algo=%s index=%s", index.algo.c_str(), index.name.c_str());
    }
  } catch (const std::exception& e) {
    log_warn("bench_build create algo failed: algo=%s index=%s error=%s",
             index.algo.c_str(),
             index.name.c_str(),
             e.what());
    return state.SkipWithError("Failed to create an algo: " + std::string(e.what()));
  }

  const auto algo_property = parse_algo_property(algo->get_preference(), index.build_param);
  if (state.thread_index() == 0) {
    log_info("bench_build algo properties: algo=%s dataset_memory_type=%d query_memory_type=%d",
             index.algo.c_str(),
             static_cast<int>(algo_property.dataset_memory_type),
             static_cast<int>(algo_property.query_memory_type));
  }

  const T* base_set      = dataset->base_set(algo_property.dataset_memory_type);
  std::size_t index_size = dataset->base_set_size();
  if (state.thread_index() == 0) {
    log_info("bench_build base_set ready: algo=%s pointer=%p index_size=%zu",
             index.algo.c_str(),
             static_cast<const void*>(base_set),
             index_size);
  }

  cuda_timer gpu_timer{algo};
  {
    nvtx_stats nvtx_stats{state};
    nvtx_case nvtx{state.name()};
    /* Note: GPU timing

    The GPU time is measured between construction and destruction of `cuda_lap` objects (`gpu_all`
    and `gpu_lap` variables) and added to the `gpu_timer` object.

    We sync with the GPU (cudaEventSynchronize) either each iteration (lifetime of the `gpu_lap`
    variable) or once per benchmark loop (lifetime of the `gpu_all` variable). The decision is

    controlled by the `no_lap_sync` argument. In either case, we need at least one sync throughout
    the benchmark loop to make sure the GPU has finished its work before we measure the total run
    time.
    */
    [[maybe_unused]] auto gpu_all = gpu_timer.lap(no_lap_sync);
    for (auto _ : state) {
      if (state.thread_index() == 0) {
        log_info(
          "bench_build iteration start: algo=%s index=%s", index.algo.c_str(), index.name.c_str());
      }
      [[maybe_unused]] auto ntx_lap = nvtx.lap();
      [[maybe_unused]] auto gpu_lap = gpu_timer.lap(!no_lap_sync);
      try {
        algo->build(base_set, index_size);
        if (state.thread_index() == 0) {
          log_info("bench_build iteration finish: algo=%s index=%s",
                   index.algo.c_str(),
                   index.name.c_str());
        }
      } catch (const std::exception& e) {
        log_warn("bench_build iteration failed: algo=%s index=%s error=%s",
                 index.algo.c_str(),
                 index.name.c_str(),
                 e.what());
        state.SkipWithError(std::string(e.what()));
      }
    }
  }
  if (gpu_timer.active()) {
    state.counters.insert({"GPU", {gpu_timer.total_time(), benchmark::Counter::kAvgIterations}});
  }
  state.counters.insert({{"index_size", index_size}});

  if (state.skipped()) { return; }
  make_sure_parent_dir_exists(index.file);
  try {
    log_info("bench_build save start: algo=%s index=%s file=%s",
             index.algo.c_str(),
             index.name.c_str(),
             index.file.c_str());
    algo->save(index.file);
    log_info("bench_build save finish: algo=%s index=%s file=%s exists=%d",
             index.algo.c_str(),
             index.name.c_str(),
             index.file.c_str(),
             file_exists(index.file) ? 1 : 0);
  } catch (const std::exception& e) {
    log_warn("bench_build save failed: algo=%s index=%s file=%s error=%s",
             index.algo.c_str(),
             index.name.c_str(),
             index.file.c_str(),
             e.what());
    state.SkipWithError("Failed to save an index: " + std::string(e.what()));
  }
}

template <typename T>
void bench_search(::benchmark::State& state,
                  const configuration::index& index,
                  std::size_t search_param_ix,
                  std::shared_ptr<const dataset<T>> dataset,
                  bool no_lap_sync)
{
  // NB: these two thread-local vars can be used within algo wrappers
  cuvs::bench::benchmark_thread_id = state.thread_index();
  cuvs::bench::benchmark_n_threads = state.threads();
  std::size_t queries_processed    = 0;

  const auto& sp_json = index.search_params[search_param_ix];

  if (state.thread_index() == 0) { dump_parameters(state, sp_json); }

  // NB: `k` and `n_queries` are guaranteed to be populated in conf.cpp
  const std::uint32_t k = sp_json["k"];
  // Amount of data processes in one go
  const std::size_t n_queries = sp_json["n_queries"];
  // Round down the query data to a multiple of the batch size to loop over full batches of data
  const std::size_t query_set_size = (dataset->query_set_size() / n_queries) * n_queries;

  if (state.thread_index() == 0) {
    log_info(
      "bench_search start: algo=%s index=%s file=%s search_param_ix=%zu threads=%d "
      "raw_queries=%zu query_set_size=%zu k=%u n_queries=%zu",
      index.algo.c_str(),
      index.name.c_str(),
      index.file.c_str(),
      search_param_ix,
      state.threads(),
      dataset->query_set_size(),
      query_set_size,
      k,
      n_queries);
  }

  if (dataset->query_set_size() < n_queries) {
    std::stringstream msg;
    msg << "Not enough queries in benchmark set. Expected " << n_queries << ", actual "
        << dataset->query_set_size();
    if (state.thread_index() == 0) { log_warn("bench_search skipped: %s", msg.str().c_str()); }
    state.SkipWithError(msg.str());
    return;
  }

  // Each thread start from a different offset, so that the queries that they process do not
  // overlap.
  std::ptrdiff_t batch_offset   = (state.thread_index() * n_queries) % query_set_size;
  std::ptrdiff_t queries_stride = state.threads() * n_queries;
  // Output is saved into a contiguous buffer (separate buffers for each thread).
  std::ptrdiff_t out_offset = 0;

  const T* query_set = nullptr;

  if (!file_exists(index.file)) {
    if (state.thread_index() == 0) {
      log_warn("bench_search missing index file: algo=%s index=%s file=%s",
               index.algo.c_str(),
               index.name.c_str(),
               index.file.c_str());
    }
    state.SkipWithError("Index file is missing. Run the benchmark in the build mode first.");
    return;
  }
  if (state.thread_index() == 0) {
    log_info("bench_search index file exists: file=%s", index.file.c_str());
  }

  /**
   * Make sure the first thread loads the algo and dataset
   */
  progress_barrier load_barrier{};
  if (load_barrier.arrive(1) == 0) {
    // algo is static to cache it between close search runs to save time on index loading
    static std::string index_file = "";
    if (index.file != index_file) {
      log_info("bench_search cache reset: previous_file=%s new_file=%s",
               index_file.c_str(),
               index.file.c_str());
      current_algo.reset();
      index_file = index.file;
    }

    std::unique_ptr<typename algo<T>::search_param> search_param;
    algo<T>* a;
    try {
      if (!current_algo || (a = dynamic_cast<algo<T>*>(current_algo.get())) == nullptr) {
        log_info("bench_search create algo start: algo=%s index=%s dim=%d",
                 index.algo.c_str(),
                 index.name.c_str(),
                 dataset->dim());
        auto ualgo =
          create_algo<T>(index.algo, dataset->distance(), dataset->dim(), index.build_param);
        a = ualgo.get();
        log_info(
          "bench_search load algo start: algo=%s file=%s", index.algo.c_str(), index_file.c_str());
        a->load(index_file);
        log_info(
          "bench_search load algo finish: algo=%s file=%s", index.algo.c_str(), index_file.c_str());
        current_algo = std::move(ualgo);
      } else {
        log_info("bench_search using cached algo: algo=%s file=%s",
                 index.algo.c_str(),
                 index_file.c_str());
      }
      log_info("bench_search create search_param start: algo=%s index=%s",
               index.algo.c_str(),
               index.name.c_str());
      search_param = create_search_param<T>(index.algo, sp_json);
      log_info("bench_search create search_param finish: algo=%s index=%s",
               index.algo.c_str(),
               index.name.c_str());
    } catch (const std::exception& e) {
      log_warn("bench_search load/create failed: algo=%s index=%s file=%s error=%s",
               index.algo.c_str(),
               index.name.c_str(),
               index_file.c_str(),
               e.what());
      state.SkipWithError("Failed to create an algo: " + std::string(e.what()));
      return;
    }

    current_algo_props =
      std::make_unique<algo_property>(std::move(parse_algo_property(a->get_preference(), sp_json)));
    log_info("bench_search algo properties: algo=%s dataset_memory_type=%d query_memory_type=%d",
             index.algo.c_str(),
             static_cast<int>(current_algo_props->dataset_memory_type),
             static_cast<int>(current_algo_props->query_memory_type));

    if (search_param->needs_dataset()) {
      try {
        log_info("bench_search set_search_dataset start: algo=%s index=%s",
                 index.algo.c_str(),
                 index.name.c_str());
        a->set_search_dataset(dataset->base_set(current_algo_props->dataset_memory_type),
                              dataset->base_set_size());
        log_info("bench_search set_search_dataset finish: algo=%s index=%s",
                 index.algo.c_str(),
                 index.name.c_str());
      } catch (const std::exception& ex) {
        log_warn("bench_search set_search_dataset failed: algo=%s index=%s error=%s",
                 index.algo.c_str(),
                 index.name.c_str(),
                 ex.what());
        state.SkipWithError("The algorithm '" + index.name +
                            "' requires the base set, but it's not available. " +
                            "Exception: " + std::string(ex.what()));
        return;
      }
    }
    try {
      log_info("bench_search set_search_param start: algo=%s index=%s",
               index.algo.c_str(),
               index.name.c_str());
      a->set_search_param(*search_param,
                          dataset->filter_bitset(current_algo_props->dataset_memory_type));
      log_info("bench_search set_search_param finish: algo=%s index=%s",
               index.algo.c_str(),
               index.name.c_str());
    } catch (const std::exception& ex) {
      log_warn("bench_search set_search_param failed: algo=%s index=%s error=%s",
               index.algo.c_str(),
               index.name.c_str(),
               ex.what());
      state.SkipWithError("An error occurred setting search parameters: " + std::string(ex.what()));
      return;
    }

    query_set = dataset->query_set(current_algo_props->query_memory_type);
    log_info("bench_search query_set acquired: algo=%s pointer=%p query_memory_type=%d",
             index.algo.c_str(),
             static_cast<const void*>(query_set),
             static_cast<int>(current_algo_props->query_memory_type));
    load_barrier.arrive(state.threads());
  } else {
    // All other threads will wait for the first thread to initialize the algo.
    load_barrier.wait(state.threads() * 2);
    // gbench ensures that all threads are synchronized at the start of the benchmark loop.
    // We are accessing shared variables (like current_algo, current_algo_probs) before the
    // benchmark loop, therefore the synchronization here is necessary.
  }
  query_set = dataset->query_set(current_algo_props->query_memory_type);

  /**
   * Each thread will manage its own outputs
   */
  using index_type                 = algo_base::index_type;
  constexpr size_t kAlignResultBuf = 64;
  size_t result_elem_count         = k * query_set_size;
  result_elem_count =
    ((result_elem_count + kAlignResultBuf - 1) / kAlignResultBuf) * kAlignResultBuf;
  auto& result_buf =
    get_result_buffer_from_global_pool(result_elem_count * (sizeof(float) + sizeof(index_type)));
  auto* neighbors_ptr =
    reinterpret_cast<index_type*>(result_buf.data(current_algo_props->query_memory_type));
  auto* distances_ptr = reinterpret_cast<float*>(neighbors_ptr + result_elem_count);
  if (state.thread_index() == 0) {
    log_info("bench_search result buffer ready: algo=%s result_elem_count=%zu memory_type=%d",
             index.algo.c_str(),
             result_elem_count,
             static_cast<int>(current_algo_props->query_memory_type));
  }

  {
    nvtx_stats nvtx_stats{state};
    nvtx_case nvtx{state.name()};

    std::unique_ptr<algo<T>> a{nullptr};
    try {
      if (state.thread_index() == 0) {
        log_info(
          "bench_search algo copy start: algo=%s index=%s", index.algo.c_str(), index.name.c_str());
      }
      dynamic_cast<algo<T>*>(current_algo.get())->copy().swap(a);
      if (state.thread_index() == 0) {
        log_info("bench_search algo copy finish: algo=%s index=%s",
                 index.algo.c_str(),
                 index.name.c_str());
      }
    } catch (const std::exception& e) {
      log_warn("bench_search algo copy failed: algo=%s index=%s error=%s",
               index.algo.c_str(),
               index.name.c_str(),
               e.what());
      state.SkipWithError("Algo::copy: " + std::string(e.what()));
      return;
    }
    // Initialize with algo, so that the timer.lap() object can sync with algo::get_sync_stream()
    cuda_timer gpu_timer{a};
    auto start = std::chrono::high_resolution_clock::now();
    {
      /* See the note above: GPU timing */
      [[maybe_unused]] auto gpu_all = gpu_timer.lap(no_lap_sync);
      for (auto _ : state) {
        [[maybe_unused]] auto ntx_lap = nvtx.lap();
        [[maybe_unused]] auto gpu_lap = gpu_timer.lap(!no_lap_sync);
        try {
          a->search(query_set + batch_offset * dataset->dim(),
                    n_queries,
                    k,
                    neighbors_ptr + out_offset * k,
                    distances_ptr + out_offset * k);
        } catch (const std::exception& e) {
          log_warn(
            "bench_search benchmark loop failed: algo=%s index=%s batch_offset=%td "
            "out_offset=%td n_queries=%zu k=%u error=%s",
            index.algo.c_str(),
            index.name.c_str(),
            batch_offset,
            out_offset,
            n_queries,
            k,
            e.what());
          state.SkipWithError("Benchmark loop: " + std::string(e.what()));
          break;
        }

        // advance to the next batch
        batch_offset = (batch_offset + queries_stride) % query_set_size;
        out_offset   = (out_offset + n_queries) % query_set_size;

        queries_processed += n_queries;
      }
    }
    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    if (state.thread_index() == 0) { state.counters.insert({{"end_to_end", duration}}); }
    state.counters.insert({"Latency", {duration, benchmark::Counter::kAvgIterations}});

    if (gpu_timer.active()) {
      state.counters.insert({"GPU", {gpu_timer.total_time(), benchmark::Counter::kAvgIterations}});
    }
  }

  state.SetItemsProcessed(queries_processed);

  // This will be the total number of queries across all threads
  state.counters.insert({{"total_queries", queries_processed}});

  if (state.skipped()) {
    if (state.thread_index() == 0) {
      log_warn("bench_search skipped before recall: algo=%s index=%s queries_processed=%zu",
               index.algo.c_str(),
               index.name.c_str(),
               queries_processed);
    }
    return;
  }

  // Each thread calculates recall on their partition of queries.
  // evaluate recall
  if (dataset->max_k() >= k && dataset->gt_maps().has_value()) {
    if (state.thread_index() == 0) {
      log_info("bench_search recall start: algo=%s index=%s max_k=%u k=%u queries_processed=%zu",
               index.algo.c_str(),
               index.name.c_str(),
               dataset->max_k(),
               k,
               queries_processed);
    }
    // gt_maps[i] is a hash map of {id, neighbor_rank} for query i
    const auto& gt_maps = dataset->gt_maps();
    result_buf.transfer_data(MemoryType::kHost, current_algo_props->query_memory_type);
    auto* neighbors_host    = reinterpret_cast<index_type*>(result_buf.data(MemoryType::kHost));
    std::size_t rows        = std::min(queries_processed, query_set_size);
    std::size_t match_count = 0;
    std::size_t total_count = 0;

    // We go through the groundtruth with same stride as the benchmark loop.
    size_t out_offset   = 0;
    size_t batch_offset = (state.thread_index() * n_queries) % query_set_size;
    // Avoid CPU oversubscription when parallelizing recall calculation loop
    int num_recall_calculation_worker_threads =
      std::thread::hardware_concurrency() / benchmark_n_threads - 1;  // -1 for the main thread
    // ensure non-negative number of workers (possible if hardware_concurrency()
    // does not return an expected value) by clamping to 0
    if (num_recall_calculation_worker_threads < 0) { num_recall_calculation_worker_threads = 0; }
    while (out_offset < rows) {
      std::vector<std::thread> recall_calculation_workers;
      recall_calculation_workers.reserve(num_recall_calculation_worker_threads);
      std::vector<std::size_t> local_match_count(num_recall_calculation_worker_threads + 1);
      std::vector<std::size_t> local_total_count(num_recall_calculation_worker_threads + 1);
      int chunk_size =
        n_queries / (num_recall_calculation_worker_threads + 1);  // +1 for the main thread
      int remainder           = n_queries % (num_recall_calculation_worker_threads + 1);
      auto recall_calculation = [&](int start, int end, int tid) -> void {
        for (int i = start; i < end; ++i) {
          size_t i_orig_idx = batch_offset + i;
          size_t i_out_idx  = out_offset + i;
          if (i_out_idx < rows) {
            auto* candidates       = neighbors_host + i_out_idx * k;
            auto [matching, total] = gt_maps->count_matches(i_orig_idx, candidates, k);
            local_match_count[tid] += matching;
            local_total_count[tid] += total;
          }
        }
      };
      // launch worker threads
      int start = 0;
      for (int tid = 0; tid < num_recall_calculation_worker_threads; tid++) {
        int end = start + chunk_size;
        if (tid < remainder) { ++end; }
        recall_calculation_workers.emplace_back(recall_calculation, start, end, tid);
        start = end;
      }
      // main thread works on last chunk
      recall_calculation(start, n_queries, num_recall_calculation_worker_threads);
      // join all worker threads
      for (auto& worker : recall_calculation_workers) {
        worker.join();
      }
      match_count += std::accumulate(local_match_count.begin(), local_match_count.end(), 0);
      total_count += std::accumulate(local_total_count.begin(), local_total_count.end(), 0);

      out_offset += n_queries;
      batch_offset = (batch_offset + queries_stride) % query_set_size;
    }
    double actual_recall = static_cast<double>(match_count) / static_cast<double>(total_count);
    if (state.thread_index() == 0) {
      log_info(
        "bench_search recall finish: algo=%s index=%s match_count=%zu total_count=%zu "
        "recall=%f",
        index.algo.c_str(),
        index.name.c_str(),
        match_count,
        total_count,
        actual_recall);
    }
    /* NOTE: recall in the throughput mode & filtering

    When filtering is enabled, `total_count` may vary between individual threads, but we still take
    the simple average across in-thread recalls. Strictly speaking, this is incorrect, but it's good
    enough under assumption that the filtering is more-or-less uniform.
    */
    state.counters.insert({"Recall", {actual_recall, benchmark::Counter::kAvgThreads}});
  }
}

inline void printf_usage()
{
  ::benchmark::PrintDefaultHelp();
  fprintf(
    stdout,
    "          [--build|--search] \n"
    "          [--force]\n"
    "          [--data_prefix=<prefix>]\n"
    "          [--index_prefix=<prefix>]\n"
    "          [--override_kv=<key:value1:value2:...:valueN>]\n"
    "          [--mode=<latency|throughput>\n"
    "          [--threads=min[:max]]\n"
    "          [--no-lap-sync]\n"
    "          <conf>.json\n"
    "\n"
    "Note the non-standard benchmark parameters:\n"
    "  --build: build mode, will build index\n"
    "  --search: search mode, will search using the built index\n"
    "            one and only one of --build and --search should be specified\n"
    "  --force: force overwriting existing index files\n"
    "  --data_prefix=<prefix>:"
    " prepend <prefix> to dataset file paths specified in the <conf>.json (default = "
    "'data/').\n"
    "  --index_prefix=<prefix>:"
    " prepend <prefix> to index file paths specified in the <conf>.json (default = "
    "'index/').\n"
    "  --override_kv=<key:value1:value2:...:valueN>:"
    " override a build/search key one or more times multiplying the number of configurations;"
    " you can use this parameter multiple times to get the Cartesian product of benchmark"
    " configs.\n"
    "  --mode=<latency|throughput>"
    " run the benchmarks in latency (accumulate times spent in each batch) or "
    " throughput (pipeline batches and measure end-to-end) mode\n"
    "  --threads=min[:max] specify the number threads to use for throughput benchmark."
    " Power of 2 values between 'min' and 'max' will be used. If only 'min' is specified,"
    " then a single test is run with 'min' threads. By default min=1, max=<num hyper"
    " threads>.\n"
    "  --no-lap-sync disable CUDA event synchronization between benchmark iterations. If a GPU"
    " algorithm has no sync with CPU, this can make the GPU processing significantly lag behind the"
    " CPU scheduling. Then this also hides the scheduling latencies and thus improves the measured"
    " throughput (QPS). Note there's a sync at the end of the benchmark loop in any case.\n");
}

template <typename T>
void register_build(std::shared_ptr<const dataset<T>> dataset,
                    std::vector<configuration::index>& indices,
                    bool force_overwrite,
                    bool no_lap_sync)
{
  for (auto& index : indices) {
    auto suf      = static_cast<std::string>(index.build_param["override_suffix"]);
    auto file_suf = suf;
    index.build_param.erase("override_suffix");
    std::replace(file_suf.begin(), file_suf.end(), '/', '-');
    index.file += file_suf;
    auto* b = ::benchmark::RegisterBenchmark(
      index.name + suf, bench_build<T>, dataset, index, force_overwrite, no_lap_sync);
    b->Unit(benchmark::kSecond);
    b->MeasureProcessCPUTime();
    b->UseRealTime();
  }
}

template <typename T>
void register_search(std::shared_ptr<const dataset<T>> dataset,
                     std::vector<configuration::index>& indices,
                     Mode metric_objective,
                     const std::vector<int>& threads,
                     bool no_lap_sync)
{
  for (auto& index : indices) {
    for (std::size_t i = 0; i < index.search_params.size(); i++) {
      auto suf = static_cast<std::string>(index.search_params[i]["override_suffix"]);
      index.search_params[i].erase("override_suffix");

      auto* b = ::benchmark::RegisterBenchmark(
                  index.name + suf, bench_search<T>, index, i, dataset, no_lap_sync)
                  ->Unit(benchmark::kMillisecond)
                  /**
                   * The following are important for getting accuracy QPS measurements on both CPU
                   * and GPU These make sure that
                   *   - `end_to_end` ~ (`Time` * `Iterations`)
                   *   - `items_per_second` ~ (`total_queries` / `end_to_end`)
                   *   - Throughput = `items_per_second`
                   */
                  ->MeasureProcessCPUTime()
                  ->UseRealTime();

      if (metric_objective == Mode::kThroughput) { b->ThreadRange(threads[0], threads[1]); }
    }
  }
}

template <typename T>
void dispatch_benchmark(std::string cmdline,
                        configuration& conf,
                        bool force_overwrite,
                        bool build_mode,
                        bool search_mode,
                        kv_series override_kv,
                        Mode metric_objective,
                        const std::vector<int>& threads,
                        bool no_lap_sync)
{
  ::benchmark::AddCustomContext("command_line", cmdline);
  for (auto [key, value] : host_info()) {
    ::benchmark::AddCustomContext(key, value);
  }
  if (cudart.found()) {
    for (auto [key, value] : cuda_info()) {
      ::benchmark::AddCustomContext(key, value);
    }
  }
  auto& dataset_conf = conf.get_dataset_conf();
  auto base_file     = dataset_conf.base_file;
  auto query_file    = dataset_conf.query_file;
  auto gt_file       = dataset_conf.groundtruth_neighbors_file;
  auto dataset =
    std::make_shared<bench::dataset<T>>(dataset_conf.name,
                                        base_file,
                                        dataset_conf.subset_first_row,
                                        dataset_conf.subset_size,
                                        query_file,
                                        dataset_conf.distance,
                                        gt_file,
                                        search_mode ? dataset_conf.filtering_rate : std::nullopt);
  ::benchmark::AddCustomContext("dataset", dataset_conf.name);
  ::benchmark::AddCustomContext("distance", dataset_conf.distance);
  std::vector<configuration::index>& indices = conf.get_indices();
  if (build_mode) {
    if (file_exists(base_file)) {
      log_info("Using the dataset file '%s'", base_file.c_str());
      ::benchmark::AddCustomContext("n_records", std::to_string(dataset->base_set_size()));
      ::benchmark::AddCustomContext("dim", std::to_string(dataset->dim()));
    } else {
      log_warn("dataset file '%s' does not exist; benchmarking index building is impossible.",
               base_file.c_str());
    }
    std::vector<configuration::index> more_indices{};
    for (auto& index : indices) {
      for (auto param : apply_overrides(index.build_param, override_kv)) {
        auto modified_index        = index;
        modified_index.build_param = param;
        more_indices.push_back(modified_index);
      }
    }
    std::swap(more_indices, indices);  // update the config in case algorithms need to access it
    register_build<T>(dataset, indices, force_overwrite, no_lap_sync);
  } else if (search_mode) {
    if (file_exists(query_file)) {
      log_info("Using the query file '%s'", query_file.c_str());
      ::benchmark::AddCustomContext("max_n_queries", std::to_string(dataset->query_set_size()));
      ::benchmark::AddCustomContext("dim", std::to_string(dataset->dim()));
      if (gt_file.has_value()) {
        if (file_exists(*gt_file)) {
          log_info("Using the ground truth file '%s'", gt_file->c_str());
          ::benchmark::AddCustomContext("max_k", std::to_string(dataset->max_k()));
        } else {
          log_warn("Ground truth file '%s' does not exist; the recall won't be reported.",
                   gt_file->c_str());
        }
      } else {
        log_warn(
          "Ground truth file is not provided; the recall won't be reported. NB: use "
          "the 'groundtruth_neighbors_file' alongside the 'query_file' key to specify the "
          "path to "
          "the ground truth in your conf.json.");
      }
    } else {
      log_warn("Query file '%s' does not exist; benchmarking search is impossible.",
               query_file.c_str());
    }
    for (auto& index : indices) {
      index.search_params = apply_overrides(index.search_params, override_kv);
    }
    register_search<T>(dataset, indices, metric_objective, threads, no_lap_sync);
  }
}

inline auto parse_bool_flag(const char* arg, const char* pat, bool& result) -> bool
{
  if (strcmp(arg, pat) == 0) {
    result = true;
    return true;
  }
  return false;
}

inline auto parse_string_flag(const char* arg, const char* pat, std::string& result) -> bool
{
  auto n = strlen(pat);
  if (strncmp(pat, arg, strlen(pat)) == 0) {
    result = arg + n + 1;
    return true;
  }
  return false;
}

inline auto run_main(int argc, char** argv) -> int
{
  bool force_overwrite        = false;
  bool build_mode             = false;
  bool search_mode            = false;
  bool no_lap_sync            = false;
  std::string data_prefix     = "data";
  std::string index_prefix    = "index";
  std::string new_override_kv = "";
  std::string mode            = "latency";
  std::string threads_arg_txt = "";
  std::vector<int> threads    = {1, -1};  // min_thread, max_thread
  kv_series override_kv{};

  char arg0_default[] = "benchmark";  // NOLINT
  char* args_default  = arg0_default;
  if (!argv) {
    argc = 1;
    argv = &args_default;
  }
  if (argc == 1) {
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
    if (parse_bool_flag(argv[i], "--force", force_overwrite) ||
        parse_bool_flag(argv[i], "--build", build_mode) ||
        parse_bool_flag(argv[i], "--search", search_mode) ||
        parse_bool_flag(argv[i], "--no-lap-sync", no_lap_sync) ||
        parse_string_flag(argv[i], "--data_prefix", data_prefix) ||
        parse_string_flag(argv[i], "--index_prefix", index_prefix) ||
        parse_string_flag(argv[i], "--mode", mode) ||
        parse_string_flag(argv[i], "--override_kv", new_override_kv) ||
        parse_string_flag(argv[i], "--threads", threads_arg_txt)) {
      if (!threads_arg_txt.empty()) {
        auto threads_arg = split(threads_arg_txt, ':');
        threads[0]       = std::stoi(threads_arg[0]);
        if (threads_arg.size() > 1) {
          threads[1] = std::stoi(threads_arg[1]);
        } else {
          threads[1] = threads[0];
        }
        threads_arg_txt = "";
      }
      if (!new_override_kv.empty()) {
        auto kvv = split(new_override_kv, ':');
        auto key = kvv[0];
        std::vector<nlohmann::json> vals{};
        for (std::size_t j = 1; j < kvv.size(); j++) {
          vals.push_back(nlohmann::json::parse(kvv[j]));
        }
        override_kv.emplace_back(key, vals);
        new_override_kv = "";
      }
      for (int j = i; j < argc - 1; j++) {
        argv[j] = argv[j + 1];
      }
      argc--;
      i--;
    }
  }

  Mode metric_objective = Mode::kLatency;
  if (mode == "throughput") { metric_objective = Mode::kThroughput; }

  int max_threads =
    (metric_objective == Mode::kThroughput) ? std::thread::hardware_concurrency() : 1;
  if (threads[1] == -1) threads[1] = max_threads;

  if (metric_objective == Mode::kLatency) {
    if (threads[0] != 1 || threads[1] != 1) {
      log_warn("Latency mode enabled. Overriding threads arg, running with single thread.");
      threads = {1, 1};
    }
  }

  if (build_mode == search_mode) {
    log_error("One and only one of --build and --search should be specified");
    printf_usage();
    return -1;
  }

  if (!conf_stream) {
    log_error("Can't open configuration file: %s", conf_path);
    return -1;
  }

  if (cudart.needed() && !cudart.found()) {
    log_warn("cudart library is not found, GPU-based indices won't work.");
  }

  auto& conf        = bench::configuration::initialize(conf_stream, data_prefix, index_prefix);
  std::string dtype = conf.get_dataset_conf().dtype;

  if (dtype == "float") {
    dispatch_benchmark<float>(cmdline,
                              conf,
                              force_overwrite,
                              build_mode,
                              search_mode,
                              override_kv,
                              metric_objective,
                              threads,
                              no_lap_sync);
  } else if (dtype == "half") {
    dispatch_benchmark<half>(cmdline,
                             conf,
                             force_overwrite,
                             build_mode,
                             search_mode,
                             override_kv,
                             metric_objective,
                             threads,
                             no_lap_sync);
  } else if (dtype == "uint8") {
    dispatch_benchmark<std::uint8_t>(cmdline,
                                     conf,
                                     force_overwrite,
                                     build_mode,
                                     search_mode,
                                     override_kv,
                                     metric_objective,
                                     threads,
                                     no_lap_sync);
  } else if (dtype == "int8") {
    dispatch_benchmark<std::int8_t>(cmdline,
                                    conf,
                                    force_overwrite,
                                    build_mode,
                                    search_mode,
                                    override_kv,
                                    metric_objective,
                                    threads,
                                    no_lap_sync);
  } else {
    log_error("datatype '%s' is not supported", dtype.c_str());
    return -1;
  }

  ::benchmark::Initialize(&argc, argv, printf_usage);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return -1;
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  // Release a possibly cached algo object, so that it cannot be alive longer than the handle
  // to a shared library it depends on (dynamic benchmark executable).
  current_algo.reset();
  current_algo_props.reset();
  reset_global_device_resources();
  return 0;
}
};  // namespace cuvs::bench

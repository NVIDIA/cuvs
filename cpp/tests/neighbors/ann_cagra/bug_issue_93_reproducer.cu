/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Reproducer for https://github.com/rapidsai/cuvs-lucene/issues/93
 *   cuvsCagraSearch returned 0 (Reason=cudaErrorInvalidValue:invalid argument)
 *
 * ROOT CAUSE:
 *   `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size)`
 *   is not thread-safe. It sets a CUDA-context-wide attribute. When two threads call it
 *   concurrently with different smem_size values, the following race occurs:
 *     1. Thread A sets max-dynamic-shared-mem to SIZE_A (larger).
 *     2. Thread B overwrites it with SIZE_B (smaller).
 *     3. Thread A launches its kernel requesting SIZE_A of shared memory,
 *        but the CUDA context now only allows SIZE_B → cudaErrorInvalidValue.
 *
 * HOW IT MANIFESTS IN cuvs-lucene:
 *   Lucene's TaskExecutor dispatches per-segment CAGRA searches to a thread pool.
 *   Each segment has a different number of vectors (e.g. 25, 26, 47), leading to
 *   different graph degrees after reduction, and therefore different smem_size values
 *   in the single-CTA search kernel. The concurrent cudaFuncSetAttribute calls race.
 *
 * REPRODUCTION STRATEGY:
 *   Build CAGRA indices with different dataset sizes (different graph degrees),
 *   then search them concurrently from separate threads, each with its own raft::resources.
 *   This mirrors the cuvs-lucene setup where each thread gets a ThreadLocal CuVSResources.
 */

#include <gtest/gtest.h>

#include "../cagra_padded_build_helpers.cuh"
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>

#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace cuvs::neighbors::cagra {

// NOLINTNEXTLINE(readability-identifier-naming)
TEST(Issue93Reproducer, ConcurrentSearchDifferentGraphDegrees)
{
  raft::resources handle;
  raft::random::RngState rng(6181234567890123459ULL);

  // Dataset sizes from REPRODUCER.md warnings (different sizes → different graph degrees).
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  std::vector<int> dataset_sizes = {25, 26, 47, 25};
  constexpr int dim              = 64;
  constexpr int top_k            = 10;

  // Build indices on the main thread (keep padded builders alive for view-based indexes).
  std::vector<cuvs::neighbors::test::padded_device_matrix_for_cagra<float>> padded_builders;
  std::vector<cagra::device_padded_index<float, uint32_t>> indices;
  for (int n_rows : dataset_sizes) {
    auto database = raft::make_device_matrix<float, int64_t>(handle, n_rows, dim);
    raft::random::uniform(
      handle, rng, database.data_handle(), n_rows * dim, -1.0F, 1.0F);  // NOLINT

    cagra::index_params ip;
    ip.metric                    = cuvs::distance::DistanceType::L2Expanded;
    ip.intermediate_graph_degree = 128;  // NOLINT
    ip.graph_degree              = 64;   // NOLINT
    ip.graph_build_params =
      graph_build_params::nn_descent_params(ip.intermediate_graph_degree, ip.metric);

    padded_builders.emplace_back(handle, raft::make_const_mdspan(database.view()));
    auto index = cagra::build(handle, ip, padded_builders.back().view);
    index.update_dataset(handle, padded_builders.back().view);
    indices.push_back(std::move(index));
  }
  raft::resource::sync_stream(handle);

  // Search concurrently from multiple threads until the first failure.
  const int num_threads = static_cast<int>(indices.size());
  std::mutex error_mutex;
  std::string first_error;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  for (int iter = 0; iter < 50 && first_error.empty(); ++iter) {
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
      threads.emplace_back([&, t, iter]() {
        raft::resources thread_handle;
        raft::random::RngState thread_rng(42ULL + static_cast<uint64_t>(t) +
                                          static_cast<uint64_t>(iter) * 1000ULL);
        try {
          auto query = raft::make_device_matrix<float, int64_t>(thread_handle, 1, dim);
          raft::random::uniform(thread_handle, thread_rng, query.data_handle(), dim, -1.0F, 1.0F);

          // Match cuvs-lucene params: Java's Panama zero-initializes the struct,
          // and SINGLE_CTA = 0 in the enum, so algo is SINGLE_CTA.
          cagra::search_params sp;
          sp.itopk_size   = top_k;
          sp.search_width = 1;
          sp.algo         = search_algo::SINGLE_CTA;

          auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(thread_handle, 1, top_k);
          auto distances = raft::make_device_matrix<float, int64_t>(thread_handle, 1, top_k);

          cagra::search(thread_handle,
                        sp,
                        indices[static_cast<size_t>(t)],
                        raft::make_const_mdspan(query.view()),
                        neighbors.view(),
                        distances.view());

          raft::resource::sync_stream(thread_handle);
        } catch (const std::exception& e) {
          std::lock_guard<std::mutex> lock(error_mutex);
          if (first_error.empty()) { first_error = e.what(); }
        }
      });
    }
    for (auto& th : threads) {
      th.join();
    }
  }

  ASSERT_TRUE(first_error.empty()) << first_error;
}

}  // namespace cuvs::neighbors::cagra

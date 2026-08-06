/*
 * TEMPORARY PR DEBUGGING
 *
 * This executable stress-tests batched silhouette score under concurrent processes in the
 * A100/CUDA-12 CI environment. Remove this entire file before merge.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/stats/silhouette_score.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream_pool.hpp>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct metric_case {
  std::string_view name;
  cuvs::distance::DistanceType metric;
};

constexpr std::array<metric_case, 4> metrics{{
  {"cosine", cuvs::distance::DistanceType::CosineExpanded},
  {"euclidean", cuvs::distance::DistanceType::L2SqrtUnexpanded},
  {"sqeuclidean", cuvs::distance::DistanceType::L2Expanded},
  {"l1", cuvs::distance::DistanceType::L1},
}};

// TEMPORARY PR DEBUGGING: Worker mode preserves the diagnostic gist workload exactly while allowing
// the parent process to create the same eight-process GPU contention within one CTest executable.
int run_worker(unsigned long seed)
{
  constexpr int64_t rows    = 1000;
  constexpr int64_t cols    = 2;
  constexpr int labels      = 2;
  constexpr int repetitions = 4;
  constexpr float tolerance = 1e-4f;
  constexpr std::array<int64_t, 3> chunks{rows, rows / 3, rows / 5};

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> centers(-1.0f, 1.0f);
  std::normal_distribution<float> noise(0.0f, 1.5f);
  std::array<std::array<float, cols>, labels> center{};
  for (auto& c : center) {
    for (auto& x : c) {
      x = centers(rng);
    }
  }
  std::vector<int64_t> order(rows);
  for (int64_t i = 0; i < rows; ++i) {
    order[i] = i;
  }
  std::shuffle(order.begin(), order.end(), rng);
  std::vector<float> X(rows * cols);
  std::vector<int> y(rows);
  for (int64_t row = 0; row < rows; ++row) {
    auto label = static_cast<int>(order[row] / (rows / labels));
    y[row]     = label;
    for (int64_t col = 0; col < cols; ++col) {
      X[row * cols + col] = center[label][col] + noise(rng);
    }
  }

  raft::resources default_handle;
  raft::resources pool_handle;
  raft::resource::set_cuda_stream_pool(pool_handle, std::make_shared<rmm::cuda_stream_pool>(4));
  auto stream = raft::resource::get_cuda_stream(default_handle);
  auto d_X    = raft::make_device_matrix<float, int64_t>(default_handle, rows, cols);
  auto d_y    = raft::make_device_vector<int, int64_t>(default_handle, rows);
  raft::update_device(d_X.data_handle(), X.data(), X.size(), stream);
  raft::update_device(d_y.data_handle(), y.data(), y.size(), stream);
  raft::resource::sync_stream(default_handle);
  auto X_view = raft::make_device_matrix_view<const float, int64_t>(d_X.data_handle(), rows, cols);
  auto y_view = raft::make_device_vector_view<const int, int64_t>(d_y.data_handle(), rows);

  bool failed = false;
  for (auto const& metric : metrics) {
    auto non_batched = cuvs::stats::silhouette_score(
      default_handle, X_view, y_view, std::nullopt, labels, metric.metric);
    for (auto const& handle :
         {std::pair{"default", &default_handle}, std::pair{"pool", &pool_handle}}) {
      for (auto chunk : chunks) {
        for (int repetition = 0; repetition < repetitions; ++repetition) {
          auto batched = cuvs::stats::silhouette_score_batched(
            *handle.second, X_view, y_view, std::nullopt, labels, chunk, metric.metric);
          if (std::abs(batched - non_batched) > tolerance) {
            failed = true;
            std::cerr << std::setprecision(10) << "seed=" << seed << " handle=" << handle.first
                      << " metric=" << metric.name << " chunk=" << chunk
                      << " repetition=" << repetition << " non-batched=" << non_batched
                      << " batched=" << batched << " difference=" << std::abs(batched - non_batched)
                      << '\n';
          }
        }
      }
    }
  }
  return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}

std::string executable_path()
{
  std::array<char, 4096> path{};
  auto length = readlink("/proc/self/exe", path.data(), path.size() - 1);
  if (length < 0) {
    std::cerr << "readlink(/proc/self/exe) failed: " << std::strerror(errno) << '\n';
    return {};
  }
  return std::string(path.data(), static_cast<std::size_t>(length));
}

// TEMPORARY PR DEBUGGING: Launch seeds 0-511 in batches of eight self-exec workers to reproduce
// the process-level concurrency from the original A100/CUDA-12 diagnostic command.
int run_orchestrator()
{
  constexpr int seed_count    = 512;
  constexpr int process_count = 8;
  auto executable             = executable_path();
  if (executable.empty()) { return EXIT_FAILURE; }

  bool failed = false;
  for (int first_seed = 0; first_seed < seed_count; first_seed += process_count) {
    std::array<pid_t, process_count> children{};
    for (int offset = 0; offset < process_count; ++offset) {
      auto seed        = first_seed + offset;
      children[offset] = fork();
      if (children[offset] == 0) {
        auto seed_string = std::to_string(seed);
        execl(executable.c_str(),
              executable.c_str(),
              "--worker",
              seed_string.c_str(),
              static_cast<char*>(nullptr));
        std::cerr << "seed=" << seed << " exec failed: " << std::strerror(errno) << '\n';
        _exit(127);
      }
      if (children[offset] < 0) {
        failed = true;
        std::cerr << "seed=" << seed << " fork failed: " << std::strerror(errno) << '\n';
      }
    }

    for (int offset = 0; offset < process_count; ++offset) {
      if (children[offset] < 0) { continue; }
      int status  = 0;
      auto waited = waitpid(children[offset], &status, 0);
      auto seed   = first_seed + offset;
      if (waited < 0) {
        failed = true;
        std::cerr << "seed=" << seed << " waitpid failed: " << std::strerror(errno) << '\n';
      } else if (WIFSIGNALED(status)) {
        failed = true;
        std::cerr << "seed=" << seed << " terminated by signal " << WTERMSIG(status) << '\n';
      } else if (!WIFEXITED(status) || WEXITSTATUS(status) != EXIT_SUCCESS) {
        failed = true;
        std::cerr << "seed=" << seed
                  << " worker exit code=" << (WIFEXITED(status) ? WEXITSTATUS(status) : -1) << '\n';
      }
    }
  }
  return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv)
{
  // TEMPORARY PR DEBUGGING: This private flag is used only by the self-exec stress harness.
  if (argc == 3 && std::string_view(argv[1]) == "--worker") {
    char* end = nullptr;
    errno     = 0;
    auto seed = std::strtoul(argv[2], &end, 10);
    if (errno != 0 || end == argv[2] || *end != '\0' || seed > 511) {
      std::cerr << "invalid worker seed: " << argv[2] << '\n';
      return EXIT_FAILURE;
    }
    return run_worker(seed);
  }
  if (argc != 1) {
    std::cerr << "usage: " << argv[0] << " [--worker SEED]\n";
    return EXIT_FAILURE;
  }
  return run_orchestrator();
}

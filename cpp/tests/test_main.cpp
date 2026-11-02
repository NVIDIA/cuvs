/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda/memory_resource>

#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

/// Returns the mode given by `--rmm_mode=<mode>` (or `--rmm_mode <mode>`), else by
/// `GTEST_CUVS_RMM_MODE`, else "async".
std::string get_rmm_mode(int argc, char** argv)
{
  char const* env_rmm_mode = std::getenv("GTEST_CUVS_RMM_MODE");  // Overridden by CLI options
  std::string rmm_mode{env_rmm_mode ? env_rmm_mode : "async"};
  // InitGoogleTest only removes gtest's own flags from argv, so `--rmm_mode` is still there.
  constexpr std::string_view flag{"--rmm_mode="};
  for (int i = 1; i < argc; ++i) {
    std::string_view const arg{argv[i]};
    if (arg.starts_with(flag)) {
      rmm_mode = arg.substr(flag.size());
    } else if (arg == "--rmm_mode" && i + 1 < argc) {
      rmm_mode = argv[++i];
    }
  }
  return rmm_mode;
}

/// Creates the device memory resource for `rmm_mode` (as in cudf::test::create_memory_resource).
cuda::mr::any_resource<cuda::mr::device_accessible> make_memory_resource(
  std::string const& rmm_mode)
{
  if (rmm_mode == "cuda") { return rmm::mr::cuda_memory_resource{}; }
  if (rmm_mode == "async") { return rmm::mr::cuda_async_memory_resource{}; }
  throw std::invalid_argument("Invalid RMM mode '" + rmm_mode + "', expected 'cuda' or 'async'");
}

}  // namespace

/**
 * @brief Entry point shared by all libcuvs gtests.
 *
 * Behaves like gtest_main, with two additions:
 *
 * - The RMM device memory resource is chosen by the command line option `--rmm_mode=<mode>` or
 *   the environment variable `GTEST_CUVS_RMM_MODE` (the option takes precedence), like cuDF's
 *   `--rmm_mode` / `GTEST_CUDF_RMM_MODE`. Supported modes are `async`
 *   (`rmm::mr::cuda_async_memory_resource`, the default) and `cuda`
 *   (`rmm::mr::cuda_memory_resource`).
 * - If the environment variable `GTEST_CUVS_MEMORY_PEAK` is set, the chosen resource is wrapped in
 *   an `rmm::mr::statistics_resource_adaptor` and the peak number of bytes allocated through RMM is
 *   printed after the tests complete. This is used by cpp/scripts/gtest_memory_usage.sh to size
 *   the ctest GPU resource PERCENT of each test.
 */
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  // Resources share ownership of their state, so `resource` and `stats` keep the ones installed as
  // the current device resource alive for the duration of the tests.
  auto resource = make_memory_resource(get_rmm_mode(argc, argv));
  rmm::mr::set_current_device_resource(resource);
  int rc{};
  if (std::getenv("GTEST_CUVS_MEMORY_PEAK")) {
    auto stats = rmm::mr::statistics_resource_adaptor{resource};
    rmm::mr::set_current_device_resource(stats);
    rc = RUN_ALL_TESTS();
    std::cout << "Peak memory usage " << stats.get_bytes_counter().peak << " bytes" << std::endl;
  } else {
    rc = RUN_ALL_TESTS();
  }
  // Release the resources (e.g. the async pool) before the CUDA runtime is torn down.
  rmm::mr::reset_current_device_resource();
  return rc;
}

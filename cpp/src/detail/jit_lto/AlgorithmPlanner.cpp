/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <iterator>
#include <memory>
#include <mutex>
#include <new>
#include <shared_mutex>
#include <string>
#include <vector>

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/nvjitlink_checker.hpp>

#include "cuda_runtime.h"
#include "nvJitLink.h"

#include <raft/core/logger.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

std::string AlgorithmPlanner::get_fragments_key() const
{
  std::string key = "";
  for (const auto& fragment : this->fragments) {
    key += fragment->get_key();
  }
  return key;
}

std::shared_ptr<AlgorithmLauncher> AlgorithmPlanner::read_cache(std::string const& launch_key) const
{
  auto& launchers = jit_cache_.launchers;
  std::shared_lock<std::shared_mutex> read_lock(jit_cache_.mutex);
  if (auto it = launchers.find(launch_key); it != launchers.end()) { return it->second; }
  return nullptr;
}

std::shared_ptr<AlgorithmLauncher> AlgorithmPlanner::get_launcher()
{
  auto& launchers = jit_cache_.launchers;
  auto launch_key = this->get_fragments_key();

  if (auto hit = read_cache(launch_key)) { return hit; }

  std::unique_lock<std::shared_mutex> write_lock(jit_cache_.mutex);
  if (auto it = launchers.find(launch_key); it != launchers.end()) { return it->second; }

  std::string log_message =
    "JIT compiling launcher for kernel: " + this->entrypoint + " and device functions: ";
  for (const auto& fragment : this->fragments) {
    log_message += std::string{fragment->get_key()} + ",";
  }
  log_message.pop_back();
  RAFT_LOG_DEBUG("%s", log_message.c_str());
  auto launcher         = this->build();
  launchers[launch_key] = launcher;
  return launcher;
}

std::shared_ptr<AlgorithmLauncher> AlgorithmPlanner::build()
{
  int device = 0;
  int major  = 0;
  int minor  = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&device));
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));

  std::string archs = "-arch=sm_" + std::to_string((major * 10 + minor));

  // Load the generated LTO IR and link them together
  nvJitLinkHandle handle;
  std::vector<const char*> lopts;
  lopts.reserve(2 + linktime_extra_options.size());
  lopts.push_back("-lto");
  lopts.push_back(archs.c_str());
  for (auto const& opt : linktime_extra_options) {
    lopts.push_back(opt.c_str());
  }
  auto result = nvJitLinkCreate(&handle, static_cast<unsigned int>(lopts.size()), lopts.data());
  check_nvjitlink_result(handle, result);

  for (const auto& frag : this->fragments) {
    frag->add_to(handle);
  }

  // Call to nvJitLinkComplete causes linker to link together all the LTO-IR
  // modules perform any optimizations and generate cubin from it.
  result = nvJitLinkComplete(handle);
  check_nvjitlink_result(handle, result);

  // get cubin from nvJitLink
  size_t cubin_size;
  result = nvJitLinkGetLinkedCubinSize(handle, &cubin_size);
  check_nvjitlink_result(handle, result);

  std::unique_ptr<char[]> cubin{new char[cubin_size]};
  result = nvJitLinkGetLinkedCubin(handle, cubin.get());
  check_nvjitlink_result(handle, result);

  result = nvJitLinkDestroy(&handle);
  RAFT_EXPECTS(result == NVJITLINK_SUCCESS, "nvJitLinkDestroy failed");

  // cubin is linked, so now load it
  cudaLibrary_t library;
  RAFT_CUDA_TRY(
    cudaLibraryLoadData(&library, cubin.get(), nullptr, nullptr, 0, nullptr, nullptr, 0));

  cudaKernel_t kernel;
  RAFT_CUDA_TRY(cudaLibraryGetKernel(&kernel, library, this->entrypoint.c_str()));

  return std::make_shared<AlgorithmLauncher>(kernel, library);
}

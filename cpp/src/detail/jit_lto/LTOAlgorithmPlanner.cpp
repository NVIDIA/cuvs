/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <new>
#include <string>
#include <vector>

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/nvjitlink_checker.hpp>

#include "cuda_runtime.h"
#include "nvJitLink.h"

#include <raft/util/cuda_rt_essentials.hpp>

std::string LTOAlgorithmPlanner::get_planner_key() const
{
  std::string key;
  for (const auto& fragment : this->fragments) {
    key += fragment->get_key();
  }
  return key;
}

std::shared_ptr<AlgorithmLauncher> LTOAlgorithmPlanner::build()
{
  int device = 0;
  int major  = 0;
  int minor  = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&device));
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));

  std::string archs = "-arch=sm_" + std::to_string((major * 10 + minor));

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

  result = nvJitLinkComplete(handle);
  check_nvjitlink_result(handle, result);

  size_t cubin_size;
  result = nvJitLinkGetLinkedCubinSize(handle, &cubin_size);
  check_nvjitlink_result(handle, result);

  std::unique_ptr<char[]> cubin{new char[cubin_size]};
  result = nvJitLinkGetLinkedCubin(handle, cubin.get());
  check_nvjitlink_result(handle, result);

  result = nvJitLinkDestroy(&handle);
  RAFT_EXPECTS(result == NVJITLINK_SUCCESS, "nvJitLinkDestroy failed");

  cudaLibrary_t library;
  RAFT_CUDA_TRY(
    cudaLibraryLoadData(&library, cubin.get(), nullptr, nullptr, 0, nullptr, nullptr, 0));

  cudaKernel_t kernel;
  RAFT_CUDA_TRY(cudaLibraryGetKernel(&kernel, library, this->entrypoint.c_str()));

  return std::make_shared<AlgorithmLauncher>(kernel, library);
}

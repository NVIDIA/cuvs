/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <cuvs/detail/jit_lto/AlgorithmLauncher.hpp>
#include <cuvs/detail/jit_lto/FragmentEntry.hpp>
#include <cuvs/detail/jit_lto/tileir_compat.hpp>

#include <raft/util/cuda_rt_essentials.hpp>

namespace cuvs::detail::jit_lto {

struct CutileModuleImage {
  const uint8_t* data;
  size_t size;
};

inline bool get_device_compute_capability(int& cc_major, int& cc_minor)
{
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) { return false; }
  if (cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) {
    return false;
  }
  if (cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
    return false;
  }
  return true;
}

/** Selects a prebuilt cubin for the device CC, or embedded TileIR when the driver can JIT it. */
inline std::optional<CutileModuleImage> resolve_cutile_module_image(
  int cc_major,
  int cc_minor,
  int driver_version,
  const std::vector<std::unique_ptr<CubinFragmentEntry>>& cubin_fragments,
  const TileIrBytecodeFragmentEntry* tileir_fragment)
{
  for (const auto& fragment : cubin_fragments) {
    if (fragment->get_cc_major() == cc_major && fragment->get_cc_minor() == cc_minor) {
      return CutileModuleImage{fragment->get_data(), fragment->get_length()};
    }
  }
  if (tileir_fragment != nullptr && tileir_fallback_available(driver_version)) {
    return CutileModuleImage{tileir_fragment->get_data(), tileir_fragment->get_length()};
  }
  return std::nullopt;
}

inline std::shared_ptr<AlgorithmLauncher> load_cutile_launcher(const CutileModuleImage& image,
                                                               const std::string& kernel_symbol)
{
  cudaLibrary_t library{};
  RAFT_CUDA_TRY(
    cudaLibraryLoadData(&library, image.data, nullptr, nullptr, 0, nullptr, nullptr, 0));

  cudaKernel_t kernel{};
  RAFT_CUDA_TRY(cudaLibraryGetKernel(&kernel, library, kernel_symbol.c_str()));

  return std::make_shared<AlgorithmLauncher>(kernel, library);
}

}  // namespace cuvs::detail::jit_lto

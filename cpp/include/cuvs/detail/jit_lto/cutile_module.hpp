/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuvs/detail/jit_lto/CutileFragmentEntry.hpp>
#include <cuvs/detail/jit_lto/tileir_compat.hpp>

#include <raft/util/cuda_rt_essentials.hpp>
#include <rtcx/algorithm_launcher.hpp>

namespace cuvs::detail::jit_lto {

struct CutileModuleImage {
  const uint8_t* data;
  size_t size;
};

/**
 * Selects the newest compatible cubin in the device's compute-capability major family.
 *
 * CUDA cubins are forward compatible across minor revisions within a major family, so an SM 8.9
 * device can load SM 8.6 SASS and an SM 12.1 device can load SM 12.0 SASS.
 */
inline const CubinFragmentEntry* find_compatible_cubin_fragment(
  int cc_major,
  int cc_minor,
  const std::vector<std::unique_ptr<CubinFragmentEntry>>& cubin_fragments)
{
  const CubinFragmentEntry* best = nullptr;
  for (const auto& fragment : cubin_fragments) {
    if (fragment->get_cc_major() != cc_major || fragment->get_cc_minor() > cc_minor) { continue; }
    if (best == nullptr || fragment->get_cc_minor() > best->get_cc_minor()) {
      best = fragment.get();
    }
  }
  return best;
}

/** Selects compatible prebuilt SASS for the device, or TileIR when the driver can JIT it. */
inline std::optional<CutileModuleImage> resolve_cutile_module_image(
  const CutileRuntimeCapabilities& capabilities,
  const std::vector<std::unique_ptr<CubinFragmentEntry>>& cubin_fragments,
  const TileIrBytecodeFragmentEntry* tileir_fragment)
{
  if (const auto* fragment = find_compatible_cubin_fragment(
        capabilities.cc_major, capabilities.cc_minor, cubin_fragments)) {
    return CutileModuleImage{fragment->get_data(), fragment->get_length()};
  }
  if (tileir_fragment != nullptr && tileir_fallback_available(capabilities.driver_version)) {
    return CutileModuleImage{tileir_fragment->get_data(), tileir_fragment->get_length()};
  }
  return std::nullopt;
}

inline bool is_expected_cutile_unavailable(cudaError_t status)
{
  switch (status) {
    case cudaErrorInvalidDeviceFunction:
    case cudaErrorInvalidPtx:
    case cudaErrorNoKernelImageForDevice:
    case cudaErrorSymbolNotFound:
    case cudaErrorUnsupportedPtxVersion:
    case cudaErrorCallRequiresNewerDriver:
    case cudaErrorSharedObjectSymbolNotFound:
    case cudaErrorSharedObjectInitFailed:
    case cudaErrorJitCompilerNotFound: return true;
    default: return false;
  }
}

/**
 * Loads a cuTile launcher, returning null for an expected module/JIT compatibility rejection.
 * Unexpected CUDA failures retain the normal RAFT exception behavior.
 */
inline std::shared_ptr<rtcx::algorithm_launcher> try_load_cutile_launcher(
  const CutileModuleImage& image, const std::string& kernel_symbol)
{
  cudaLibrary_t library{};
  auto load_status =
    cudaLibraryLoadData(&library, image.data, nullptr, nullptr, 0, nullptr, nullptr, 0);
  if (load_status != cudaSuccess) {
    if (is_expected_cutile_unavailable(load_status)) { return nullptr; }
    RAFT_CUDA_TRY(load_status);
  }

  cudaKernel_t kernel{};
  load_status = cudaLibraryGetKernel(&kernel, library, kernel_symbol.c_str());
  if (load_status != cudaSuccess) {
    RAFT_CUDA_TRY(cudaLibraryUnload(library));
    if (is_expected_cutile_unavailable(load_status)) { return nullptr; }
    RAFT_CUDA_TRY(load_status);
  }

  return std::make_shared<rtcx::algorithm_launcher>(kernel, library);
}

}  // namespace cuvs::detail::jit_lto

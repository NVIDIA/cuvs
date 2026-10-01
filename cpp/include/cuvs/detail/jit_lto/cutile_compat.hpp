/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifndef CUVS_CUTILE_ENABLED
#define CUVS_CUTILE_ENABLED 0
#endif

#include <cuda_runtime.h>

#include <cuvs/detail/jit_lto/cutile_arch_tags.hpp>

namespace cuvs::detail::jit_lto {

/** Runtime/device properties that determine cuTile image selection and launch eligibility. */
struct CutileRuntimeCapabilities {
  int device;
  int cc_major;
  int cc_minor;
};

inline bool query_current_cutile_runtime_capabilities(CutileRuntimeCapabilities& capabilities)
{
  if (cudaGetDevice(&capabilities.device) != cudaSuccess) { return false; }
  if (cudaDeviceGetAttribute(&capabilities.cc_major,
                             cudaDevAttrComputeCapabilityMajor,
                             capabilities.device) != cudaSuccess) {
    return false;
  }
  if (cudaDeviceGetAttribute(&capabilities.cc_minor,
                             cudaDevAttrComputeCapabilityMinor,
                             capabilities.device) != cudaSuccess) {
    return false;
  }
  return true;
}

/** Minimum CUDA runtime version (from cudaRuntimeGetVersion) for cuTile integration. */
inline constexpr int kMinCutileRuntimeVersion = 13000;

inline constexpr bool library_built_with_cutile()
{
#if CUVS_CUTILE_ENABLED
  return true;
#else
  return false;
#endif
}

inline bool runtime_cuda13_or_newer()
{
  int runtime_version = 0;
  if (cudaRuntimeGetVersion(&runtime_version) != cudaSuccess) { return false; }
  return runtime_version >= kMinCutileRuntimeVersion;
}

/** True when this build embeds cuTile artifacts and the runtime is CUDA 13+. */
inline bool cutile_integration_enabled()
{
  return library_built_with_cutile() && runtime_cuda13_or_newer();
}

/** True when this build embeds compatible SASS for the device. */
inline bool has_compatible_embedded_cubin_for_arch(int cc_major, int cc_minor)
{
  return is_embedded_cubin_arch(cc_major, cc_minor) ||
         (can_use_sm86_compat_cubin(cc_major, cc_minor) && is_embedded_cubin_arch(8, 6));
}

/**
 * True when a cuTile launch may be attempted for the given device: cuTile is enabled, the runtime
 * is CUDA 13+, and an exact cubin exists, except that SM89 may use the embedded SM86 cubin.
 */
#if CUVS_CUTILE_ENABLED
inline bool cutile_launch_available_for_arch(int cc_major, int cc_minor)
{
  return runtime_cuda13_or_newer() && has_compatible_embedded_cubin_for_arch(cc_major, cc_minor);
}
#else
inline constexpr bool cutile_launch_available_for_arch(int, int) { return false; }
#endif

#if CUVS_CUTILE_ENABLED
inline bool cutile_launch_available_on_current_device()
{
  CutileRuntimeCapabilities capabilities{};
  if (!query_current_cutile_runtime_capabilities(capabilities)) { return false; }
  return cutile_launch_available_for_arch(capabilities.cc_major, capabilities.cc_minor);
}
#else
/** Compile-time false when cuTile is not built; use in if constexpr to skip cuTile-only paths. */
inline constexpr bool cutile_launch_available_on_current_device() { return false; }
#endif

}  // namespace cuvs::detail::jit_lto

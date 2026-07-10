/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifndef CUVS_CUTILE_ENABLED
#define CUVS_CUTILE_ENABLED 0
#endif

#include <cuda.h>
#include <cuda_runtime.h>

#include <cuvs/detail/jit_lto/cutile_arch_tags.hpp>

namespace cuvs::detail::jit_lto {

/** Minimum CUDA driver version (from cudaDriverGetVersion) for TileIR JIT of embedded bytecode. */
inline constexpr int kMinTileIrJitDriverVersion = 13010;  // CUDA 13.1 / driver >= 590.44

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

/** True when this build embeds a prebuilt cubin for the given compute capability. */
inline bool has_embedded_cubin_for_arch(int cc_major, int cc_minor)
{
  return is_embedded_cubin_arch(cc_major, cc_minor);
}

/** True when the driver can JIT-compile embedded TileIR bytecode at load time. */
inline bool tileir_fallback_available(int driver_version)
{
  return driver_version >= kMinTileIrJitDriverVersion;
}

/**
 * True when a cuTile launch may be attempted for the given device: cuTile is enabled, the runtime
 * is CUDA 13+, and either a matching embedded cubin exists (no driver JIT required) or the driver
 * can JIT the embedded TileIR bytecode fallback.
 */
#if CUVS_CUTILE_ENABLED
inline bool cutile_launch_available_for_arch(int cc_major, int cc_minor, int driver_version)
{
  if (!runtime_cuda13_or_newer()) { return false; }
  if (has_embedded_cubin_for_arch(cc_major, cc_minor)) { return true; }
  return tileir_fallback_available(driver_version);
}
#else
inline constexpr bool cutile_launch_available_for_arch(int, int, int) { return false; }
#endif

inline bool query_driver_version(int& driver_version)
{
  return cudaDriverGetVersion(&driver_version) == cudaSuccess;
}

inline bool query_current_device_arch(int& cc_major, int& cc_minor)
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

#if CUVS_CUTILE_ENABLED
inline bool cutile_launch_available_on_current_device()
{
  int cc_major       = 0;
  int cc_minor       = 0;
  int driver_version = 0;
  if (!query_current_device_arch(cc_major, cc_minor)) { return false; }
  if (!query_driver_version(driver_version)) { return false; }
  return cutile_launch_available_for_arch(cc_major, cc_minor, driver_version);
}
#else
/** Compile-time false when cuTile is not built; use in if constexpr to skip cuTile-only paths. */
inline constexpr bool cutile_launch_available_on_current_device() { return false; }
#endif

}  // namespace cuvs::detail::jit_lto

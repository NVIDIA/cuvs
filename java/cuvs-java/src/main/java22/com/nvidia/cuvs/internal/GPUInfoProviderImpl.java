/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal;

import static com.nvidia.cuvs.internal.common.LinkerHelper.C_INT;
import static com.nvidia.cuvs.internal.common.Util.checkCudaError;
import static com.nvidia.cuvs.internal.common.Util.cudaGetDeviceProperties;
import static com.nvidia.cuvs.internal.panama.headers_h.*;
import static com.nvidia.cuvs.internal.panama.headers_h.cudaMemGetInfo;
// Import from the leaf binding class (headers_h extends headers_h_1 …); it inherits every symbol,
// so this is robust to jextract shifting symbols across the split classes on regeneration.

import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.CuVSResourcesInfo;
import com.nvidia.cuvs.GPUInfo;
import com.nvidia.cuvs.GPUInfoProvider;
import com.nvidia.cuvs.internal.panama.cudaDeviceProp;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

public class GPUInfoProviderImpl implements GPUInfoProvider {

  // Lazy initialization for list of available GPUs.
  private static class AvailableGpuInitializer {

    // Available GPUs are initialized only once when first accessed.
    // This is assumed to be invariant for the lifetime of the program.
    static final List<GPUInfo> AVAILABLE_GPUS = getAvailableGpusInfo();

    private static List<GPUInfo> getAvailableGpusInfo() {
      try (var localArena = Arena.ofConfined()) {

        MemorySegment numGpus = localArena.allocate(C_INT);
        int returnValue = cudaGetDeviceCount(numGpus);
        checkCudaError(returnValue, "cudaGetDeviceCount");

        int numGpuCount = numGpus.get(C_INT, 0);
        List<GPUInfo> gpuInfoArr = new ArrayList<GPUInfo>();

        MemorySegment deviceProp = cudaDeviceProp.allocate(localArena);

        for (int i = 0; i < numGpuCount; i++) {
          returnValue = cudaGetDeviceProperties(deviceProp, i);
          checkCudaError(returnValue, "cudaGetDeviceProperties");

          GPUInfo gpuInfo =
              new GPUInfo(
                  i,
                  cudaDeviceProp.name(deviceProp).getString(0),
                  cudaDeviceProp.totalGlobalMem(deviceProp),
                  cudaDeviceProp.major(deviceProp),
                  cudaDeviceProp.minor(deviceProp),
                  cudaDeviceProp.asyncEngineCount(deviceProp) > 0,
                  cudaDeviceProp.concurrentKernels(deviceProp) > 0);

          gpuInfoArr.add(gpuInfo);
        }
        return gpuInfoArr;
      }
    }
  }

  private static boolean hasMinimumCapability(GPUInfo gpuInfo) {
    return gpuInfo.computeCapabilityMajor() > GPUInfoProvider.MIN_COMPUTE_CAPABILITY_MAJOR
        || (gpuInfo.computeCapabilityMajor() == GPUInfoProvider.MIN_COMPUTE_CAPABILITY_MAJOR
            && gpuInfo.computeCapabilityMinor() >= GPUInfoProvider.MIN_COMPUTE_CAPABILITY_MINOR);
  }

  @Override
  public List<GPUInfo> availableGPUs() {
    return AvailableGpuInitializer.AVAILABLE_GPUS;
  }

  @Override
  public List<GPUInfo> compatibleGPUs() {
    List<GPUInfo> compatibleGPUs = new ArrayList<>();
    long minDeviceMemoryInBytes = 1024L * 1024L * GPUInfoProvider.MIN_DEVICE_MEMORY_IN_MB;
    for (GPUInfo gpuInfo : AvailableGpuInitializer.AVAILABLE_GPUS) {
      if (hasMinimumCapability(gpuInfo)
          && gpuInfo.totalDeviceMemoryInBytes() >= minDeviceMemoryInBytes) {
        compatibleGPUs.add(gpuInfo);
      }
    }
    return compatibleGPUs;
  }

  @Override
  public CuVSResourcesInfo getCurrentInfo(CuVSResources resources) {
    try (var localArena = Arena.ofConfined()) {
      var deviceIdPtr = localArena.allocate(C_INT);
      checkCudaError(cudaGetDevice(deviceIdPtr), "cudaGetDevice");
      var currentDeviceId = deviceIdPtr.get(C_INT, 0);

      if (resources.deviceId() != currentDeviceId) {
        checkCudaError(cudaSetDevice(resources.deviceId()), "cudaSetDevice");
      }

      MemorySegment freeMemoryPtr = localArena.allocate(size_t);
      MemorySegment totalMemoryPtr = localArena.allocate(size_t);
      checkCudaError(cudaMemGetInfo(freeMemoryPtr, totalMemoryPtr), "cudaMemGetInfo");

      if (resources.deviceId() != currentDeviceId) {
        checkCudaError(cudaSetDevice(currentDeviceId), "cudaSetDevice");
      }

      return new CuVSResourcesInfo(freeMemoryPtr.get(size_t, 0), totalMemoryPtr.get(size_t, 0));
    }
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

/**
 * A Dataset implementation backed by host (CPU) memory.
 */
public interface CuVSHostMatrix extends CuVSMatrix {
  int get(int row, int col);

  default CuVSDeviceMatrix toDevice(CuVSResources resources) {
    try (var builder = CuVSMatrix.deviceBuilder(resources, size(), columns(), dataType())) {
      var deviceMatrix = builder.build();
      try {
        toDevice(deviceMatrix, resources);
        return deviceMatrix;
      } catch (RuntimeException | Error failure) {
        // The caller owns the destination only after the copy succeeds.
        try (deviceMatrix) {
          throw failure;
        }
      }
    }
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

/**
 * A Dataset implementation backed by host (CPU) memory.
 */
public interface CuVSHostMatrix extends CuVSMatrix {
  int get(int row, int col);

  default CuVSDeviceMatrix toDevice(CuVSResources resources) {
    CuVSDeviceMatrix deviceMatrix =
        MatrixBuilderLifecycle.buildAndClose(
            CuVSMatrix.deviceBuilder(resources, size(), columns(), dataType()));
    try {
      toDevice(deviceMatrix, resources);
      return deviceMatrix;
    } catch (RuntimeException | Error failure) {
      try {
        deviceMatrix.close();
      } catch (RuntimeException | Error closeFailure) {
        if (failure != closeFailure) {
          failure.addSuppressed(closeFailure);
        }
      }
      throw failure;
    }
  }
}

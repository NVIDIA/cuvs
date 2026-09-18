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
    var deviceMatrix = CuVSMatrix.deviceBuilder(resources, size(), columns(), dataType()).build();
    toDevice(deviceMatrix, resources);
    return deviceMatrix;
  }
}

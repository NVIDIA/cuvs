/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

/**
 * A Dataset implementation backed by device (GPU) memory.
 */
public interface CuVSDeviceMatrix extends CuVSMatrix {

  /**
   * Returns a new host matrix with data from this device matrix.
   * The returned host matrix will need to be managed by the caller, which will be
   * responsible to call {@link CuVSMatrix#close()} to free its resources when done.
   */
  default CuVSHostMatrix toHost() {
    var hostMatrix = CuVSMatrix.hostBuilder(size(), columns(), dataType()).build();
    toHost(hostMatrix);
    return hostMatrix;
  }
}

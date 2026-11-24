/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    try (var builder = CuVSMatrix.hostBuilder(size(), columns(), dataType())) {
      var hostMatrix = builder.build();
      try {
        toHost(hostMatrix);
        return hostMatrix;
      } catch (RuntimeException | Error failure) {
        // The caller owns the destination only after the copy succeeds.
        try (hostMatrix) {
          throw failure;
        }
      }
    }
  }
}

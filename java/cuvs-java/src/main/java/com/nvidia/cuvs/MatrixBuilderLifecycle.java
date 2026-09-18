/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import java.util.Objects;

/** Ownership-aware finalization shared by the public matrix conversion defaults. */
final class MatrixBuilderLifecycle {

  private MatrixBuilderLifecycle() {}

  static <T extends CuVSMatrix> T buildAndClose(CuVSMatrix.Builder<T> builder) {
    T matrix;
    try {
      matrix = Objects.requireNonNull(builder.build(), "Matrix builder must not return null");
    } catch (RuntimeException | Error operationFailure) {
      closeAndSuppress(builder, operationFailure);
      throw operationFailure;
    }

    try {
      builder.close();
      return matrix;
    } catch (RuntimeException | Error builderCloseFailure) {
      closeAndSuppress(matrix, builderCloseFailure);
      throw builderCloseFailure;
    }
  }

  private static void closeAndSuppress(AutoCloseable resource, Throwable failure) {
    try {
      resource.close();
    } catch (RuntimeException | Error closeFailure) {
      if (failure != closeFailure) {
        failure.addSuppressed(closeFailure);
      }
    } catch (Exception closeFailure) {
      // CuVSMatrix and its Builder narrow close() to unchecked failures. Keep this guard so an
      // unusual AutoCloseable implementation cannot replace the original failure.
      if (failure != closeFailure) {
        failure.addSuppressed(closeFailure);
      }
    }
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CuVSMatrix;
import java.io.IOException;
import java.util.Objects;

/* Ownership-aware matrix construction shared by the Lucene ingestion paths. */
final class MatrixBuilderLifecycle {

  private MatrixBuilderLifecycle() {}

  @FunctionalInterface
  interface BuildOperation<T extends CuVSMatrix> {
    T build(CuVSMatrix.Builder<T> builder) throws Throwable;
  }

  static <T extends CuVSMatrix> T build(CuVSMatrix.Builder<T> builder, BuildOperation<T> operation)
      throws IOException {
    T matrix = null;
    Throwable operationFailure = null;
    try {
      matrix =
          Objects.requireNonNull(operation.build(builder), "Matrix builder must not return null");
    } catch (Throwable failure) {
      operationFailure = failure;
    }

    Throwable cleanupFailure = closeResource(builder, null);
    if (cleanupFailure != null && matrix != null) {
      cleanupFailure = closeResource(matrix, cleanupFailure);
    }
    Throwable failure = addFailure(operationFailure, cleanupFailure);
    if (failure != null) {
      throw Utils.handleThrowable(failure);
    }
    return matrix;
  }

  private static Throwable closeResource(AutoCloseable resource, Throwable failure) {
    try {
      resource.close();
    } catch (Throwable closeFailure) {
      return addFailure(failure, closeFailure);
    }
    return failure;
  }

  private static Throwable addFailure(Throwable primary, Throwable secondary) {
    if (secondary == null) {
      return primary;
    }
    if (primary == null) {
      return secondary;
    }
    if (primary != secondary) {
      primary.addSuppressed(secondary);
    }
    return primary;
  }
}

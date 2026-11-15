/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CuVSHostMatrix;
import com.nvidia.cuvs.CuVSMatrix;
import java.io.IOException;
import java.util.function.Supplier;
import org.apache.lucene.util.Accountable;
import org.apache.lucene.util.IOConsumer;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.RamUsageEstimator;

/**
 * Tracks the compact primary host input while one field is filled, built, and written.
 * Each writer uses one instance, with allocation scopes run sequentially by one caller at a time.
 *
 * <p>This is payload accounting, not an allocator measurement or memory limit. It excludes
 * upper-layer inputs, adjacency matrices, GPU workspace, and other temporary build storage.
 * Lucene's cached IndexWriter accounting does not poll this value during flushes or merges.
 */
final class HostInputMemory implements Accountable {
  private static final long SHALLOW_BYTES =
      RamUsageEstimator.shallowSizeOfInstance(HostInputMemory.class);

  private final InfoStream infoStream;
  private final String component;
  private final String segment;
  private long inputBytes;

  HostInputMemory(InfoStream infoStream, String component, String segment) {
    this.infoStream = infoStream;
    this.component = component;
    this.segment = segment;
  }

  /** The action must close the built dataset, or transfer it to an index that it closes. */
  void withMatrix(
      String field,
      long rows,
      long columns,
      CuVSMatrix.DataType type,
      IOConsumer<CuVSMatrix.Builder<CuVSHostMatrix>> buildAndWrite)
      throws IOException {
    long bytes = payloadBytes(rows, columns, type);
    withAllocation(field, bytes, () -> CuVSMatrix.hostBuilder(rows, columns, type), buildAndWrite);
  }

  // The factory boundary lets lifecycle tests inject allocation/cleanup failures without native
  // RAM.
  void withAllocation(
      String field,
      long bytes,
      Supplier<CuVSMatrix.Builder<CuVSHostMatrix>> allocate,
      IOConsumer<CuVSMatrix.Builder<CuVSHostMatrix>> buildAndWrite)
      throws IOException {
    try (CuVSMatrix.Builder<CuVSHostMatrix> builder = allocate.get()) {
      // Built-in host builders allocate the complete compact matrix before returning.
      inputBytes = bytes;
      if (infoStream.isEnabled(component)) {
        infoStream.message(
            component,
            "primary_host_input_bytes=" + bytes + " segment=" + segment + " field=" + field);
      }
      buildAndWrite.accept(builder);
    } finally {
      // Scope completion is not proof of deallocation if native cleanup itself failed.
      inputBytes = 0;
    }
  }

  static long payloadBytes(long rows, long columns, CuVSMatrix.DataType type) {
    if (rows < 0 || columns < 0) {
      throw new IllegalArgumentException("Matrix dimensions must be nonnegative");
    }
    return Math.multiplyExact(Math.multiplyExact(rows, columns), type.bytes());
  }

  @Override
  public long ramBytesUsed() {
    return SHALLOW_BYTES + inputBytes;
  }
}

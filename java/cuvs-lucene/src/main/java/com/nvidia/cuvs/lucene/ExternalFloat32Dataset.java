/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CuVSHostMatrix;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.spi.CuVSProvider;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * A dense float32 dataset backed by caller-owned native memory.
 *
 * <p>The memory is wrapped, not copied. The caller must keep the segment's scope alive until index
 * writing has completed. A shared scope is required because the accelerated writer may read the
 * dataset concurrently while serializing Lucene's flat-vector file.
 *
 * <p>This is an expert-only input for the unsorted, single-segment native-buffering path. Row
 * {@code i} must correspond exactly to Lucene vector ordinal and document ID {@code i}; gaps,
 * reordered documents, and multiple vector fields are rejected by the writer. The caller is also
 * responsible for ensuring that every raw vector component is finite; unlike ordinary Lucene field
 * ingestion, this path deliberately does not scan the payload one float at a time.
 */
public final class ExternalFloat32Dataset {

  private final MemorySegment memorySegment;
  private final CuVSHostMatrix matrix;
  private final int rows;
  private final int dimensions;

  private ExternalFloat32Dataset(
      MemorySegment memorySegment, CuVSHostMatrix matrix, int rows, int dimensions) {
    this.memorySegment = memorySegment;
    this.matrix = matrix;
    this.rows = rows;
    this.dimensions = dimensions;
  }

  /**
   * Wraps a contiguous native-memory region containing row-major float32 values without copying.
   */
  public static ExternalFloat32Dataset fromMemorySegment(
      MemorySegment memorySegment, int rows, int dimensions) {
    Objects.requireNonNull(memorySegment, "memorySegment");
    if (rows <= 0) {
      throw new IllegalArgumentException("rows must be positive");
    }
    if (dimensions <= 0) {
      throw new IllegalArgumentException("dimensions must be positive");
    }
    if (ByteOrder.nativeOrder() != ByteOrder.LITTLE_ENDIAN) {
      throw new UnsupportedOperationException(
          "External float32 datasets currently require a little-endian host");
    }
    if (!memorySegment.isNative()) {
      throw new IllegalArgumentException("memorySegment must be backed by native memory");
    }
    if (!memorySegment.isReadOnly()) {
      throw new IllegalArgumentException(
          "memorySegment must be read-only while the graph and flat writers consume it");
    }
    if (Math.floorMod(memorySegment.address(), Float.BYTES) != 0) {
      throw new IllegalArgumentException("memorySegment address must be float-aligned");
    }
    if (!memorySegment.scope().isAlive()) {
      throw new IllegalArgumentException("memorySegment scope is not alive");
    }
    Thread accessProbe = Thread.ofPlatform().unstarted(() -> {});
    if (!memorySegment.isAccessibleBy(accessProbe)) {
      throw new IllegalArgumentException(
          "memorySegment must have a shared scope so the flat writer can access it");
    }

    long expectedBytes =
        Math.multiplyExact(Math.multiplyExact((long) rows, dimensions), Float.BYTES);
    if (memorySegment.byteSize() != expectedBytes) {
      throw new IllegalArgumentException(
          "memorySegment byte size ("
              + memorySegment.byteSize()
              + ") must equal rows * dimensions * 4 ("
              + expectedBytes
              + ")");
    }

    final CuVSMatrix wrapped;
    try {
      wrapped =
          (CuVSMatrix)
              CuVSProvider.provider()
                  .newNativeMatrixBuilder()
                  .invokeExact(memorySegment, rows, dimensions, CuVSMatrix.DataType.FLOAT);
    } catch (Throwable t) {
      if (t instanceof Error error) {
        throw error;
      }
      if (t instanceof RuntimeException runtimeException) {
        throw runtimeException;
      }
      throw new IllegalStateException("Unable to wrap the external float32 dataset", t);
    }
    if (!(wrapped instanceof CuVSHostMatrix hostMatrix)) {
      wrapped.close();
      throw new IllegalStateException("Native-memory dataset factory did not return a host matrix");
    }
    if (hostMatrix.size() != rows
        || hostMatrix.columns() != dimensions
        || hostMatrix.dataType() != CuVSMatrix.DataType.FLOAT) {
      hostMatrix.close();
      throw new IllegalStateException(
          "Native-memory dataset factory returned an unexpected matrix shape or data type");
    }
    return new ExternalFloat32Dataset(memorySegment, hostMatrix, rows, dimensions);
  }

  public MemorySegment memorySegment() {
    return memorySegment;
  }

  public CuVSHostMatrix matrix() {
    return matrix;
  }

  public int rows() {
    return rows;
  }

  public int dimensions() {
    return dimensions;
  }

  @Override
  public String toString() {
    return "ExternalFloat32Dataset[rows=" + rows + ", dimensions=" + dimensions + "]";
  }
}

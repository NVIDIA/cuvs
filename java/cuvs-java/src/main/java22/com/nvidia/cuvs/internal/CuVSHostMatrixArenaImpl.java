/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal;

import com.nvidia.cuvs.internal.common.Util;
import java.lang.foreign.Arena;
import java.lang.foreign.SequenceLayout;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.atomic.AtomicReference;

/**
 * A Dataset implementation backed by host (CPU) memory.
 * Memory is allocated and managed by the implementation, via a Java shared {@link Arena}
 */
public class CuVSHostMatrixArenaImpl extends CuVSHostMatrixImpl {
  private final AtomicReference<Arena> arenaReference;

  public CuVSHostMatrixArenaImpl(long size, long columns, DataType dataType) {
    this(
        size,
        columns,
        -1,
        -1,
        dataType,
        valueLayoutFromType(dataType),
        sequenceLayoutFromType(size, columns, -1, dataType),
        Arena.ofShared());
  }

  public CuVSHostMatrixArenaImpl(
      long size, long columns, int rowStride, int columnStride, DataType dataType) {
    this(
        size,
        columns,
        rowStride,
        columnStride,
        dataType,
        valueLayoutFromType(dataType),
        sequenceLayoutFromType(size, columns, rowStride, dataType),
        Arena.ofShared());
  }

  private CuVSHostMatrixArenaImpl(
      long size,
      long columns,
      int rowStride,
      int columnStride,
      DataType dataType,
      ValueLayout valueLayout,
      SequenceLayout layout,
      Arena arena) {
    super(
        // Not zero-filled: every producer of these matrices writes all of it (see
        // HostMatrixBuilder.build for rows a caller never adds).
        Util.allocateUninitialized(arena, layout.byteSize(), layout.byteAlignment()),
        size,
        columns,
        rowStride,
        columnStride,
        dataType,
        valueLayout,
        layout);

    if (columnStride != -1) {
      throw new UnsupportedOperationException(
          "Stridden columns are currently not supported; columnStride must be equal to -1");
    }

    this.arenaReference = new AtomicReference<>(arena);
  }

  @Override
  public void close() {
    var arena = arenaReference.getAndSet(null);
    if (arena != null) {
      arena.close();
    }
  }
}

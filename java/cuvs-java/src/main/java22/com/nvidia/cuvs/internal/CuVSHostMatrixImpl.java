/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal;

import static com.nvidia.cuvs.internal.common.Util.prepareTensor;
import static com.nvidia.cuvs.internal.panama.headers_h.*;

import com.nvidia.cuvs.CuVSDeviceMatrix;
import com.nvidia.cuvs.CuVSHostMatrix;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.RowView;
import java.lang.foreign.*;
import java.lang.invoke.VarHandle;
import java.util.Locale;

/**
 * A Dataset implementation backed by host (CPU) memory.
 */
public class CuVSHostMatrixImpl extends CuVSMatrixBaseImpl implements CuVSHostMatrix {
  protected final VarHandle accessor$vh;

  private final int rowStride;
  private final int columnStride;
  private final long rowBytes;
  private final long rowSize;
  private final long elementSize;

  public CuVSHostMatrixImpl(
      MemorySegment memorySegment, long size, long columns, DataType dataType) {
    this(
        memorySegment,
        size,
        columns,
        -1,
        -1,
        dataType,
        valueLayoutFromType(dataType),
        sequenceLayoutFromType(size, columns, -1, dataType));
  }

  public CuVSHostMatrixImpl(
      MemorySegment memorySegment,
      long size,
      long columns,
      int rowStride,
      int columnStride,
      DataType dataType) {
    this(
        memorySegment,
        size,
        columns,
        rowStride,
        columnStride,
        dataType,
        valueLayoutFromType(dataType),
        sequenceLayoutFromType(size, columns, rowStride, dataType));
  }

  protected CuVSHostMatrixImpl(
      MemorySegment memorySegment,
      long size,
      long columns,
      int rowStride,
      int columnStride,
      DataType dataType,
      ValueLayout valueLayout,
      MemoryLayout sequenceLayout) {
    super(memorySegment, dataType, valueLayout, size, columns);

    if (rowStride > 0 && rowStride < columns) {
      throw new IllegalArgumentException("Row stride cannot be less than the number of columns");
    }

    this.rowStride = rowStride;
    this.columnStride = columnStride;

    this.elementSize = valueLayout.byteSize();
    this.rowSize = rowStride > 0 ? rowStride * elementSize : columns * elementSize;
    this.rowBytes = columns * elementSize;

    this.accessor$vh = sequenceLayout.varHandle(MemoryLayout.PathElement.sequenceElement());
  }

  @Override
  public RowView getRow(long index) {
    assert (index < size)
        : String.format(Locale.ROOT, "Index out of bound ([%d], size [%d])", index, size);

    return new SliceRowView(
        memorySegment.asSlice(index * rowSize, rowBytes),
        columns,
        valueLayout,
        dataType,
        elementSize);
  }

  @Override
  public void toArray(int[][] array) {
    assert (array.length >= size)
        : String.format(
            Locale.ROOT,
            "Input array is not large enough (required: [%d], actual [%d])",
            size,
            array.length);
    assert (array.length == 0 || array[0].length >= columns)
        : String.format(
            Locale.ROOT,
            "Input array is not wide enough (required: [%d], actual [%d])",
            columns,
            array[0].length);
    assert dataType == DataType.INT || dataType == DataType.UINT
        : String.format(
            Locale.ROOT, "Input array is of the wrong type for dataType [%s]", dataType.toString());

    for (int r = 0; r < size; ++r) {
      MemorySegment.copy(memorySegment, valueLayout, r * rowSize, array[r], 0, (int) columns);
    }
  }

  @Override
  public void toArray(float[][] array) {
    assert (array.length >= size)
        : String.format(
            Locale.ROOT,
            "Input array is not large enough (required: [%d], actual [%d])",
            size,
            array.length);
    assert (array.length == 0 || array[0].length >= columns)
        : String.format(
            Locale.ROOT,
            "Input array is not wide enough (required: [%d], actual [%d])",
            columns,
            array[0].length);
    assert dataType == DataType.FLOAT
        : String.format(
            Locale.ROOT, "Input array is of the wrong type for dataType [%s]", dataType.toString());

    for (int r = 0; r < size; ++r) {
      MemorySegment.copy(memorySegment, valueLayout, r * rowSize, array[r], 0, (int) columns);
    }
  }

  @Override
  public void toArray(byte[][] array) {
    assert (array.length >= size)
        : String.format(
            Locale.ROOT,
            "Input array is not large enough (required: [%d], actual [%d])",
            size,
            array.length);
    assert (array.length == 0 || array[0].length >= columns)
        : String.format(
            Locale.ROOT,
            "Input array is not wide enough (required: [%d], actual [%d])",
            columns,
            array[0].length);
    assert dataType == DataType.BYTE
        : String.format(
            Locale.ROOT, "Input array is of the wrong type for dataType [%s]", dataType.toString());

    for (int r = 0; r < size; ++r) {
      MemorySegment.copy(memorySegment, valueLayout, r * rowSize, array[r], 0, (int) columns);
    }
  }

  @Override
  public CuVSHostMatrix toHost() {
    return new CuVSHostMatrixDelegate(this);
  }

  @Override
  public void toHost(CuVSHostMatrix hostMatrix) {
    var targetMatrix = (CuVSMatrixInternal) hostMatrix;

    if (targetMatrix.columns() != this.columns || targetMatrix.size() != this.size) {
      throw new IllegalArgumentException(
          "Source and target matrices must have the same dimensions");
    }
    if (targetMatrix.dataType() != this.dataType) {
      throw new IllegalArgumentException("Source and target matrices must have the same dataType");
    }

    if (this.rowStride <= 0 && targetMatrix.rowStride() <= 0) {
      // copy whole matrix
      MemorySegment.copy(this.memorySegment, 0L, targetMatrix.memorySegment(), 0L, size * rowSize);
    } else {
      // copy row-by-row
      long targetRowSize =
          targetMatrix.rowStride() > 0
              ? targetMatrix.rowStride() * elementSize
              : columns * elementSize;
      for (int r = 0; r < size; ++r) {
        MemorySegment.copy(
            this.memorySegment,
            r * rowSize,
            targetMatrix.memorySegment(),
            r * targetRowSize,
            rowBytes);
      }
    }
  }

  @Override
  public void toDevice(CuVSDeviceMatrix deviceMatrix, CuVSResources cuVSResources) {
    copyMatrix(this, (CuVSMatrixInternal) deviceMatrix, cuVSResources);
  }

  @Override
  public void close() {}

  @Override
  public int get(int row, int col) {
    var rowPitch = rowStride > 0 ? rowStride : columns;
    return (int) accessor$vh.get(memorySegment, 0L, (long) row * rowPitch + col);
  }

  @Override
  public long rowStride() {
    return rowStride;
  }

  @Override
  public MemorySegment toTensor(Arena arena) {
    // DLPack strides are in elements. An unset column stride means adjacent elements, which is a
    // stride of 1; passing it through as -1 would make every consumer reject the tensor as not
    // row-major.
    var strides =
        rowStride > 0 ? new long[] {rowStride, columnStride > 0 ? columnStride : 1} : null;
    return prepareTensor(
        arena, memorySegment, new long[] {size, columns}, strides, code(), bits(), kDLCPU());
  }

  private static class CuVSHostMatrixDelegate implements CuVSHostMatrix, CuVSMatrixInternal {
    private final CuVSHostMatrixImpl hostMatrix;

    public CuVSHostMatrixDelegate(CuVSHostMatrixImpl cuVSHostMatrix) {
      this.hostMatrix = cuVSHostMatrix;
    }

    @Override
    public int get(int row, int col) {
      return hostMatrix.get(row, col);
    }

    @Override
    public long size() {
      return hostMatrix.size();
    }

    @Override
    public long columns() {
      return hostMatrix.columns();
    }

    @Override
    public DataType dataType() {
      return hostMatrix.dataType();
    }

    @Override
    public RowView getRow(long row) {
      return hostMatrix.getRow(row);
    }

    @Override
    public void toArray(int[][] array) {
      hostMatrix.toArray(array);
    }

    @Override
    public void toArray(float[][] array) {
      hostMatrix.toArray(array);
    }

    @Override
    public void toArray(byte[][] array) {
      hostMatrix.toArray(array);
    }

    @Override
    public void toHost(CuVSHostMatrix hostMatrix) {
      this.hostMatrix.toHost(hostMatrix);
    }

    @Override
    public CuVSHostMatrix toHost() {
      return this;
    }

    @Override
    public void toDevice(CuVSDeviceMatrix deviceMatrix, CuVSResources cuVSResources) {
      hostMatrix.toDevice(deviceMatrix, cuVSResources);
    }

    @Override
    public MemorySegment memorySegment() {
      return hostMatrix.memorySegment();
    }

    @Override
    public ValueLayout valueLayout() {
      return hostMatrix.valueLayout();
    }

    @Override
    public long rowStride() {
      return hostMatrix.rowStride();
    }

    @Override
    public MemorySegment toTensor(Arena arena) {
      return hostMatrix.toTensor(arena);
    }

    @Override
    public void close() {
      // Do nothing
    }
  }
}

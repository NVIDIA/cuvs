/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal;

import static com.nvidia.cuvs.internal.common.Util.*;
import static com.nvidia.cuvs.internal.panama.headers_h.*;

import com.nvidia.cuvs.*;
import com.nvidia.cuvs.internal.common.PinnedMemoryBuffer;
import com.nvidia.cuvs.internal.panama.DLManagedTensor;
import com.nvidia.cuvs.internal.panama.DLTensor;
import java.lang.foreign.*;

public class CuVSDeviceMatrixImpl extends CuVSMatrixBaseImpl implements CuVSDeviceMatrix {

  private interface RowAccessStrategy {
    RowView getRow(long row);
  }

  private final CuVSResources resources;

  private final long rowStride;
  private final long columnStride;
  private final long rowSize;
  private final long valueByteSize;

  private final RowAccessStrategy rowAccessStrategy;

  private class BufferedRowAccessStrategy implements RowAccessStrategy {
    private long bufferedMatrixRowStart = 0;
    private long bufferedMatrixRowEnd = 0;

    @Override
    public RowView getRow(long row) {
      try (var access = resources.access()) {
        var hostBuffer = CuVSResourcesImpl.getHostBuffer(access);
        long rowBytes = columns * valueByteSize;
        if (row < bufferedMatrixRowStart || row >= bufferedMatrixRowEnd) {
          var endRow = Math.min(row + (PinnedMemoryBuffer.CHUNK_BYTES / rowBytes), size);
          populateBuffer(access, row, endRow, hostBuffer);
          bufferedMatrixRowStart = row;
          bufferedMatrixRowEnd = endRow;
        }
        var startRow = row - bufferedMatrixRowStart;
        // populateBuffer packs rows (no row stride) into the host buffer.
        return new SliceRowView(
            hostBuffer.asSlice(startRow * rowBytes, rowBytes),
            columns,
            valueLayout,
            dataType,
            valueByteSize);
      }
    }
  }

  private class DirectRowAccessStrategy implements RowAccessStrategy {
    @Override
    public RowView getRow(long row) {
      try (var access = resources.access()) {
        var memorySegment = Arena.ofAuto().allocate(columns * valueByteSize);
        populateBuffer(access, row, row + 1, memorySegment);
        return new SliceRowView(memorySegment, columns, valueLayout, dataType, valueByteSize);
      }
    }
  }

  protected CuVSDeviceMatrixImpl(
      CuVSResources resources,
      MemorySegment deviceMemorySegment,
      long size,
      long columns,
      DataType dataType,
      ValueLayout valueLayout) {
    this(resources, deviceMemorySegment, size, columns, -1, -1, dataType, valueLayout);
  }

  protected CuVSDeviceMatrixImpl(
      CuVSResources resources,
      MemorySegment deviceMemorySegment,
      long size,
      long columns,
      long rowStride,
      long columnStride,
      DataType dataType,
      ValueLayout valueLayout) {
    super(deviceMemorySegment, dataType, valueLayout, size, columns);
    this.resources = resources;
    this.rowStride = rowStride;
    this.columnStride = columnStride;

    this.valueByteSize = valueLayout.byteSize();
    this.rowSize = rowStride > 0 ? rowStride * valueByteSize : columns * valueByteSize;
    if (rowSize > PinnedMemoryBuffer.CHUNK_BYTES) {
      // The shared buffer is too small for this row size, use a direct access strategy
      this.rowAccessStrategy = new DirectRowAccessStrategy();
    } else {
      this.rowAccessStrategy = new BufferedRowAccessStrategy();
    }
  }

  @Override
  public long rowStride() {
    return rowStride;
  }

  @Override
  public MemorySegment toTensor(Arena arena) {
    var strides = rowStride >= 0 ? new long[] {rowStride, columnStride} : null;
    return prepareTensor(
        arena, memorySegment, new long[] {size, columns}, strides, code(), bits(), kDLCUDA());
  }

  private void populateBuffer(
      CuVSResources.ScopedAccess resourceAccess,
      long startRow,
      long endRow,
      MemorySegment bufferAddress) {
    try (var localArena = Arena.ofConfined()) {
      var rowCount = endRow - startRow;

      MemorySegment sliceManagedTensor = DLManagedTensor.allocate(localArena);
      DLManagedTensor.dl_tensor(sliceManagedTensor, DLTensor.allocate(localArena));

      checkCuVSError(
          cuvsMatrixSliceRows(0, toTensor(localArena), startRow, endRow, sliceManagedTensor),
          "cuvsMatrixSliceRows");
      assert DLTensor.shape(DLManagedTensor.dl_tensor(sliceManagedTensor)).get(C_LONG, 0)
          == rowCount;
      assert DLTensor.shape(DLManagedTensor.dl_tensor(sliceManagedTensor)).getAtIndex(C_LONG, 1)
          == columns;

      MemorySegment bufferTensor =
          prepareTensor(
              localArena, bufferAddress, new long[] {rowCount, columns}, code(), bits(), kDLCPU());

      try {
        checkCuVSError(
            cuvsMatrixCopy(resourceAccess.handle(), sliceManagedTensor, bufferTensor),
            "cuvsMatrixCopy");
        checkCuVSError(cuvsStreamSync(resourceAccess.handle()), "cuvsStreamSync");
      } finally {
        // cuvsMatrixSliceRows heap-allocates the slice's shape (and strides); release them.
        releaseManagedTensor(sliceManagedTensor);
      }
    }
  }

  @Override
  public RowView getRow(long row) {
    return rowAccessStrategy.getRow(row);
  }

  @Override
  public void toArray(int[][] array) {
    assert dataType == DataType.INT || dataType == DataType.UINT;
    copyRowsTo(array);
  }

  @Override
  public void toArray(float[][] array) {
    assert dataType == DataType.FLOAT;
    copyRowsTo(array);
  }

  @Override
  public void toArray(byte[][] array) {
    assert dataType == DataType.BYTE;
    copyRowsTo(array);
  }

  /**
   * Copies every row into {@code array[row]}, staging as many rows at a time as fit in the pinned
   * host buffer: one slice, one copy and one stream sync per chunk rather than per row.
   */
  private void copyRowsTo(Object[] array) {
    assert (array.length >= size) : "Input array is not large enough";
    long rowBytes = columns * valueByteSize;
    try (var access = resources.access();
        var localArena = Arena.ofConfined()) {
      // A row too large for the pinned buffer goes through a pageable one, one row at a time.
      boolean fitsPinned = rowBytes <= PinnedMemoryBuffer.CHUNK_BYTES;
      var hostBuffer =
          fitsPinned ? CuVSResourcesImpl.getHostBuffer(access) : localArena.allocate(rowBytes);
      long chunkRows = fitsPinned ? PinnedMemoryBuffer.CHUNK_BYTES / rowBytes : 1;
      for (long start = 0; start < size; start += chunkRows) {
        long end = Math.min(start + chunkRows, size);
        populateBuffer(access, start, end, hostBuffer);
        for (long r = start; r < end; ++r) {
          MemorySegment.copy(
              hostBuffer, valueLayout, (r - start) * rowBytes, array[(int) r], 0, (int) columns);
        }
      }
    }
  }

  @Override
  public void toHost(CuVSHostMatrix targetMatrix) {
    copyMatrix(this, (CuVSMatrixInternal) targetMatrix, resources);
  }

  @Override
  public CuVSDeviceMatrix toDevice(CuVSResources resources) {
    return new CuVSDeviceMatrixDelegate(this);
  }

  @Override
  public void toDevice(CuVSDeviceMatrix targetMatrix, CuVSResources cuVSResources) {
    copyMatrix(this, (CuVSMatrixInternal) targetMatrix, cuVSResources);
  }

  @Override
  public void close() {}

  private static class CuVSDeviceMatrixDelegate implements CuVSDeviceMatrix, CuVSMatrixInternal {
    private final CuVSDeviceMatrixImpl deviceMatrix;

    private CuVSDeviceMatrixDelegate(CuVSDeviceMatrixImpl deviceMatrix) {
      this.deviceMatrix = deviceMatrix;
    }

    @Override
    public long size() {
      return deviceMatrix.size();
    }

    @Override
    public long columns() {
      return deviceMatrix.columns();
    }

    @Override
    public DataType dataType() {
      return deviceMatrix.dataType();
    }

    @Override
    public RowView getRow(long row) {
      return deviceMatrix.getRow(row);
    }

    @Override
    public void toArray(int[][] array) {
      deviceMatrix.toArray(array);
    }

    @Override
    public void toArray(float[][] array) {
      deviceMatrix.toArray(array);
    }

    @Override
    public void toArray(byte[][] array) {
      deviceMatrix.toArray(array);
    }

    @Override
    public void toHost(CuVSHostMatrix hostMatrix) {
      deviceMatrix.toHost(hostMatrix);
    }

    @Override
    public void toDevice(CuVSDeviceMatrix targetMatrix, CuVSResources cuVSResources) {
      deviceMatrix.toDevice(targetMatrix, cuVSResources);
    }

    @Override
    public CuVSDeviceMatrix toDevice(CuVSResources cuVSResources) {
      return this;
    }

    @Override
    public MemorySegment memorySegment() {
      return deviceMatrix.memorySegment();
    }

    @Override
    public ValueLayout valueLayout() {
      return deviceMatrix.valueLayout();
    }

    @Override
    public long rowStride() {
      return deviceMatrix.rowStride();
    }

    @Override
    public MemorySegment toTensor(Arena arena) {
      return deviceMatrix.toTensor(arena);
    }

    @Override
    public void close() {
      // Do nothing
    }
  }
}

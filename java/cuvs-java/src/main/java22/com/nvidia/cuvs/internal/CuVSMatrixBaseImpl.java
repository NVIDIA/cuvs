/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal;

import static com.nvidia.cuvs.internal.common.LinkerHelper.C_CHAR;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_FLOAT;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_INT;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_SHORT;
import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.panama.headers_h.*;

import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.internal.panama.DLDataType;
import com.nvidia.cuvs.internal.panama.DLDevice;
import com.nvidia.cuvs.internal.panama.DLManagedTensor;
import com.nvidia.cuvs.internal.panama.DLTensor;
import java.lang.foreign.*;
import java.util.Locale;

abstract class CuVSMatrixBaseImpl implements CuVSMatrixInternal {
  protected final MemorySegment memorySegment;
  protected final DataType dataType;
  protected final ValueLayout valueLayout;
  protected final long size;
  protected final long columns;

  protected CuVSMatrixBaseImpl(
      MemorySegment memorySegment,
      DataType dataType,
      ValueLayout valueLayout,
      long size,
      long columns) {
    this.memorySegment = memorySegment;
    this.dataType = dataType;
    this.valueLayout = valueLayout;
    this.size = size;
    this.columns = columns;
  }

  @Override
  public String toString() {
    return String.format("%dx%d %s @ 0x%016X", size, columns, dataType, memorySegment.address());
  }

  protected static void copyMatrix(
      CuVSMatrixInternal sourceMatrix, CuVSMatrixInternal targetMatrix, CuVSResources resources) {
    if (targetMatrix.columns() != sourceMatrix.columns()
        || targetMatrix.size() != sourceMatrix.size()) {
      throw new IllegalArgumentException(
          "Source and target matrices must have the same dimensions");
    }
    if (targetMatrix.dataType() != sourceMatrix.dataType()) {
      throw new IllegalArgumentException("Source and target matrices must have the same dataType");
    }

    try (var localArena = Arena.ofConfined()) {
      var targetTensor = targetMatrix.toTensor(localArena);

      try (var resourceAccess = resources.access()) {
        var cuvsRes = resourceAccess.handle();
        var sourceTensor = sourceMatrix.toTensor(localArena);
        checkCuVSError(cuvsMatrixCopy(cuvsRes, sourceTensor, targetTensor), "cuvsMatrixCopy");
        checkCuVSError(cuvsStreamSync(cuvsRes), "cuvsStreamSync");
      }
    }
  }

  @Override
  public long size() {
    return size;
  }

  @Override
  public long columns() {
    return columns;
  }

  @Override
  public DataType dataType() {
    return dataType;
  }

  @Override
  public MemorySegment memorySegment() {
    return memorySegment;
  }

  @Override
  public ValueLayout valueLayout() {
    return valueLayout;
  }

  protected static ValueLayout valueLayoutFromType(DataType dataType) {
    return switch (dataType) {
      case FLOAT -> C_FLOAT;
      case HALF -> C_SHORT;
      case INT, UINT -> C_INT;
      case BYTE -> C_CHAR;
    };
  }

  protected static SequenceLayout sequenceLayoutFromType(
      long size, long columns, int rowStride, DataType dataType) {
    var elements = rowStride > 0 ? rowStride * size : columns * size;
    return MemoryLayout.sequenceLayout(elements, valueLayoutFromType(dataType))
        .withByteAlignment(32);
  }

  /**
   * Creates a {@link CuVSMatrix} from data and infos from a {@link DLManagedTensor}
   *
   * @param dlManagedTensor a {@link MemorySegment} representing the source DLManagedTensor
   * @param resources       {@link CuVSResources} to allocate the resulting matrix
   * @return a {@link CuVSMatrix} encapsulating the same data as the input {@link DLManagedTensor}
   */
  public static CuVSMatrix fromTensor(MemorySegment dlManagedTensor, CuVSResources resources) {
    var dlTensor = DLManagedTensor.dl_tensor(dlManagedTensor);
    var dlDevice = DLTensor.device(dlTensor);

    var deviceType = DLDevice.device_type(dlDevice);

    var data = DLTensor.data(dlTensor);
    if (data.equals(MemorySegment.NULL)) {
      throw new IllegalArgumentException("[data] must not be NULL");
    }

    var ndim = DLTensor.ndim(dlTensor);
    if (ndim != 2) {
      throw new IllegalArgumentException("CuVSMatrix only supports 2D data");
    }

    var dtype = DLTensor.dtype(dlTensor);
    var code = DLDataType.code(dtype);
    var bits = DLDataType.bits(dtype);

    final DataType dataType = dataTypeFromTensor(code, bits);

    var shape = DLTensor.shape(dlTensor);
    if (shape.equals(MemorySegment.NULL)) {
      throw new IllegalArgumentException("[shape] must not be NULL");
    }

    var rows = shape.get(int64_t, 0);
    var cols = shape.getAtIndex(int64_t, 1);

    if (deviceType == kDLCUDA()) {
      var strides = DLTensor.strides(dlTensor);
      if (strides.equals(MemorySegment.NULL)) {
        return new CuVSDeviceMatrixImpl(
            resources, data, rows, cols, dataType, valueLayoutFromType(dataType));
      } else {
        var rowStride = strides.get(int64_t, 0);
        var colStride = strides.getAtIndex(int64_t, 1);
        return new CuVSDeviceMatrixImpl(
            resources,
            data,
            rows,
            cols,
            rowStride,
            colStride,
            dataType,
            valueLayoutFromType(dataType));
      }
    } else if (deviceType == kDLCPU()) {
      return new CuVSHostMatrixImpl(data, rows, cols, dataType);
    } else {
      throw new IllegalArgumentException("Unsupported device type: " + deviceType);
    }
  }

  private static DataType dataTypeFromTensor(byte code, byte bits) {
    final DataType dataType;
    if (code == kDLUInt() && bits == 32) {
      dataType = DataType.UINT;
    } else if (code == kDLInt() && bits == 32) {
      dataType = DataType.INT;
    } else if (code == kDLFloat() && bits == 32) {
      dataType = DataType.FLOAT;
    } else if (code == kDLFloat() && bits == 16) {
      dataType = DataType.HALF;
    } else if ((code == kDLInt() || code == kDLUInt()) && bits == 8) {
      dataType = DataType.BYTE;
    } else {
      throw new IllegalArgumentException(
          String.format(Locale.ROOT, "Unsupported data type (code=%d, bits=%d)", code, bits));
    }
    return dataType;
  }
}

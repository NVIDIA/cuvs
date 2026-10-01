/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal;

import static com.nvidia.cuvs.internal.panama.headers_h.*;

import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.internal.panama.DLManagedTensor;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Internal interface for {@link CuVSMatrix}, for shared functionality that
 * require/expose Panama types or internal cuvs types.
 */
public interface CuVSMatrixInternal extends CuVSMatrix {

  MemorySegment memorySegment();

  ValueLayout valueLayout();

  /**
   * Size (in bits) for the element type of this matrix
   */
  default int bits() {
    return (int) (valueLayout().byteSize() * 8);
  }

  /**
   * DLTensor data type {@code code} for the element type of this matrix
   */
  default int code() {
    return code(dataType());
  }

  static int code(DataType dataType) {
    return switch (dataType) {
      case FLOAT, HALF -> kDLFloat();
      case INT -> kDLInt();
      case UINT, BYTE -> kDLUInt();
    };
  }

  long rowStride();

  /**
   * Creates a {@link DLManagedTensor} representing the matrix data and shape, to be
   * passed to the CuVS C API.
   * @param arena The Arena to use to allocate DL data structures
   * @return a {@link MemorySegment} for the newly allocated DLManagedTensor
   */
  MemorySegment toTensor(Arena arena);
}

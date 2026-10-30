/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal.common;

import static com.nvidia.cuvs.internal.common.LinkerHelper.C_POINTER;
import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsRMMAlloc;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsRMMFree;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * A closeable handle for RMM allocations that can be used with try-with-resources.
 * This class manages the lifecycle of a memory allocation made with NVIDIA RMM.
 * It ensures that the allocated memory is properly released when no longer needed.
 */
public class CloseableRMMAllocation implements CloseableHandle {

  private final long cuvsResourceHandle;
  private long numBytes;
  private MemorySegment pointer;

  private CloseableRMMAllocation(long cuvsResourceHandle, long numBytes, MemorySegment pointer) {
    this.cuvsResourceHandle = cuvsResourceHandle;
    this.numBytes = numBytes;
    this.pointer = pointer;
  }

  /**
   * Copy constructor transfers the ownership of the argument's MemorySegment to this object.
   */
  public CloseableRMMAllocation(CloseableRMMAllocation other) {
    this.cuvsResourceHandle = other.cuvsResourceHandle;
    this.numBytes = other.numBytes;
    this.pointer = other.release();
  }

  public static CloseableRMMAllocation allocateRMMSegment(long cuvsResourceHandle, long numBytes) {
    try (var localArena = Arena.ofConfined()) {
      MemorySegment datasetMemorySegment = localArena.allocate(C_POINTER);
      checkCuVSError(
          cuvsRMMAlloc(cuvsResourceHandle, datasetMemorySegment, numBytes), "cuvsRMMAlloc");
      return new CloseableRMMAllocation(
          cuvsResourceHandle, numBytes, datasetMemorySegment.get(C_POINTER, 0));
    }
  }

  @Override
  public MemorySegment handle() {
    return pointer;
  }

  private MemorySegment release() {
    var oldPointer = pointer;
    pointer = MemorySegment.NULL;
    numBytes = 0;
    return oldPointer;
  }

  private boolean mustClose() {
    return pointer != MemorySegment.NULL;
  }

  @Override
  public void close() {
    if (mustClose()) {
      checkCuVSError(cuvsRMMFree(cuvsResourceHandle, pointer, numBytes), "cuvsRMMFree");
      pointer = MemorySegment.NULL;
    }
  }

  public static CloseableRMMAllocation EMPTY = new CloseableRMMAllocation(0, 0, MemorySegment.NULL);
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal.common;

import static com.nvidia.cuvs.internal.common.CloseableRMMAllocation.allocateRMMSegment;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_INT_BYTE_SIZE;
import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.common.Util.checkCudaError;
import static com.nvidia.cuvs.internal.common.Util.getStream;
import static com.nvidia.cuvs.internal.common.Util.prepareTensor;
import static com.nvidia.cuvs.internal.panama.headers_h.BITMAP;
import static com.nvidia.cuvs.internal.panama.headers_h.BITSET;
import static com.nvidia.cuvs.internal.panama.headers_h.NO_FILTER;
import static com.nvidia.cuvs.internal.panama.headers_h.cudaMemcpyAsync;
import static com.nvidia.cuvs.internal.panama.headers_h.cudaMemcpyHostToDevice;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsStreamSync;
import static com.nvidia.cuvs.internal.panama.headers_h.kDLCUDA;
import static com.nvidia.cuvs.internal.panama.headers_h.kDLUInt;

import com.nvidia.cuvs.internal.BitsetPacker;
import com.nvidia.cuvs.internal.panama.cuvsFilter;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.BitSet;

/** Builds the {@code cuvsFilter} a search or merge reads its prefilter from. */
public final class Prefilters {

  private static final ValueLayout.OfLong LITTLE_ENDIAN_LONG =
      ValueLayout.JAVA_LONG.withOrder(ByteOrder.LITTLE_ENDIAN);

  private Prefilters() {}

  /**
   * Fills {@code filter} in and returns the device allocation backing it, which the caller has to
   * keep open until the call that reads the filter returns.
   *
   * <p>A {@code null} {@code bitsets} produces a NO_FILTER and an empty allocation. A single bitset
   * produces a BITSET, shared by every query, and several produce a BITMAP with one bitset per
   * query. The bitsets are packed as described in {@link BitsetPacker}: a {@code null} entry
   * accepts every row, and an entry with no bits set rejects every row.
   */
  public static CloseableRMMAllocation fill(
      long cuvsRes, Arena arena, MemorySegment filter, BitSet[] bitsets, long numRows) {
    if (bitsets == null) {
      cuvsFilter.type(filter, NO_FILTER());
      cuvsFilter.addr(filter, 0);
      return CloseableRMMAllocation.EMPTY;
    }

    if (bitsets.length == 0) {
      throw new IllegalArgumentException("Expected at least one bitset, or null for no filter");
    }
    long[] packed = BitsetPacker.pack(bitsets, numRows);
    long words = BitsetPacker.words32(numRows * bitsets.length);
    long bytes = C_INT_BYTE_SIZE * words;
    // The GPU reads the words little-endian. Written little-endian, the packed longs hold the same
    // bytes as the 32-bit words, and cover at least as many bytes as cuVS reads.
    MemorySegment hostWords = arena.allocate((long) Long.BYTES * packed.length, Long.BYTES);
    MemorySegment.copy(packed, 0, hostWords, LITTLE_ENDIAN_LONG, 0, packed.length);

    var deviceWords = allocateRMMSegment(cuvsRes, bytes);
    try {
      // Call the binding directly: Util.cudaMemcpyAsync reports a failed copy as an AssertionError.
      checkCudaError(
          cudaMemcpyAsync(
              deviceWords.handle(), hostWords, bytes, cudaMemcpyHostToDevice(), getStream(cuvsRes)),
          "cudaMemcpyAsync");
      // The host words live in the caller's arena, so wait for the copy rather than rely on it
      // outliving the stream work.
      checkCuVSError(cuvsStreamSync(cuvsRes), "cuvsStreamSync");

      MemorySegment filterTensor =
          prepareTensor(arena, deviceWords.handle(), new long[] {words}, kDLUInt(), 32, kDLCUDA());
      cuvsFilter.type(filter, bitsets.length == 1 ? BITSET() : BITMAP());
      cuvsFilter.addr(filter, filterTensor.address());
      return deviceWords;
    } catch (Throwable t) {
      try {
        deviceWords.close();
      } catch (Exception closeError) {
        t.addSuppressed(closeError);
      }
      throw t;
    }
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal;

import static com.nvidia.cuvs.internal.common.CloseableRMMAllocation.allocateRMMSegment;
import static com.nvidia.cuvs.internal.common.Util.CudaMemcpyKind.HOST_TO_DEVICE;
import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.common.Util.cudaMemcpyAsync;
import static com.nvidia.cuvs.internal.common.Util.getStream;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsStreamSync;

import com.nvidia.cuvs.FilterBitsetHandle;
import com.nvidia.cuvs.internal.common.CloseableRMMAllocation;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Device-backed implementation of {@link FilterBitsetHandle}.
 *
 * <h3>Two-level caching</h3>
 * <ul>
 *   <li><strong>Host level</strong> – the packed {@code long[]} arrays are immutable after
 *       construction and shared safely across threads.</li>
 *   <li><strong>Device level</strong> – a single shared device allocation is uploaded once on
 *       first use (lazy, double-checked locking) and reused by all threads thereafter.</li>
 * </ul>
 */
public final class FilterBitsetHandleImpl implements FilterBitsetHandle {

  /** Device-side allocation for the combined bitset, shared across all threads. */
  static final class DeviceData {
    final CloseableRMMAllocation combinedBitsetDP;
    final long combinedWords; // number of uint32 words in the combined bitset

    DeviceData(CloseableRMMAllocation combinedBitsetDP, long combinedWords) {
      this.combinedBitsetDP = combinedBitsetDP;
      this.combinedWords = combinedWords;
    }

    void close() {
      try {
        combinedBitsetDP.close();
      } catch (Exception ignored) {
      }
    }
  }

  // Host-side immutable data.
  final long[] combinedLongs;

  // Shared device allocation — uploaded once, visible to all threads via volatile.
  private volatile DeviceData sharedDeviceData;
  private final Object uploadLock = new Object();
  private volatile boolean closed = false;

  public FilterBitsetHandleImpl(long[] combinedLongs) {
    this.combinedLongs = combinedLongs;
  }

  /**
   * Returns the shared device allocation for this filter, uploading on first call (lazy,
   * thread-safe via double-checked locking).
   *
   * @param cuvsRes the native cuvsResources handle for the calling thread
   */
  DeviceData getOrUpload(long cuvsRes) {
    if (closed) throw new IllegalStateException("FilterBitsetHandle has been closed");
    DeviceData data = sharedDeviceData;
    if (data != null) return data;
    synchronized (uploadLock) {
      data = sharedDeviceData;
      if (data != null) return data;
      data = upload(cuvsRes);
      sharedDeviceData = data; // volatile write: happens-before all subsequent reads
    }
    return data;
  }

  private DeviceData upload(long cuvsRes) {
    long combinedBitsetBytes = (long) combinedLongs.length * Long.BYTES;
    CloseableRMMAllocation combinedBitsetDP = allocateRMMSegment(cuvsRes, combinedBitsetBytes);

    var stream = getStream(cuvsRes);
    // Host arena must outlive the stream sync that confirms the H2D copy.
    try (var arena = Arena.ofConfined()) {
      MemorySegment hostBitset = arena.allocate(combinedBitsetBytes, Long.BYTES);
      MemorySegment.copy(
          combinedLongs, 0, hostBitset, ValueLayout.JAVA_LONG, 0, combinedLongs.length);
      cudaMemcpyAsync(
          combinedBitsetDP.handle(), hostBitset, combinedBitsetBytes, HOST_TO_DEVICE, stream);

      checkCuVSError(cuvsStreamSync(cuvsRes), "cuvsStreamSync in FilterBitsetHandle.upload");
    }
    // Stream sync has returned — device memory is fully populated. uint32 words = long count * 2.
    return new DeviceData(combinedBitsetDP, (long) combinedLongs.length * 2);
  }

  /** Marks this handle closed and releases the shared device allocation. */
  @Override
  public void close() {
    closed = true;
    DeviceData data;
    synchronized (uploadLock) {
      data = sharedDeviceData;
      sharedDeviceData = null;
    }
    if (data != null) data.close();
  }
}

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

import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.FilterBitsetHandle;
import com.nvidia.cuvs.internal.common.CloseableRMMAllocation;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

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

  // A process-lifetime resources that owns every filter bitset's device allocation.
  //
  // cuvsRMMAlloc/cuvsRMMFree route through the resources' per-resources workspace memory resource
  // (raft::resource::get_workspace_resource_ref) and its CUDA stream. RMM requires deallocate() to
  // run on the *same* memory resource that allocated the pointer, so the free must use the same
  // resources that uploaded -- we cannot simply free against whatever resources happens to be valid
  // (and callers configure per-resources workspace pools, so the workspace MRs are genuinely
  // distinct). Binding the allocation to the short-lived per-query resources passed to search
  // crashes: that resources can be torn down (e.g. on segment reader close) while a handle is still
  // cached in the shared FilterBitsetCache, leaving the free with a dangling resources. A dedicated,
  // never-closed resources keeps alloc and free on one always-valid workspace MR.
  private static final Object FILTER_RESOURCES_LOCK = new Object();
  private static volatile CuVSResources filterResources;

  private static CuVSResources filterResources() {
    CuVSResources r = filterResources;
    if (r == null) {
      synchronized (FILTER_RESOURCES_LOCK) {
        r = filterResources;
        if (r == null) {
          try {
            r = CuVSResources.create();
          } catch (Throwable t) {
            throw new RuntimeException("Failed to create resources for filter bitset device memory", t);
          }
          filterResources = r;
        }
      }
    }
    return r;
  }

  // Host-side immutable data.
  final long[] combinedLongs;

  // Shared device allocation — uploaded once, visible to all threads via volatile.
  private volatile DeviceData sharedDeviceData;
  private final Object uploadLock = new Object();

  // Reference count. Starts at 1 for the initial reference held by the owner and released by
  // close(); every concurrent user holds an additional reference via tryIncRef()/decRef(). The
  // device allocation is freed when the count reaches 0, so a close() concurrent with an in-flight
  // search only decrements and cannot free memory still in use.
  private final AtomicInteger refCount = new AtomicInteger(1);
  // Guards close() so the initial reference is released at most once (AutoCloseable is idempotent).
  private final AtomicBoolean closed = new AtomicBoolean(false);

  public FilterBitsetHandleImpl(long[] combinedLongs) {
    this.combinedLongs = combinedLongs;
  }

  @Override
  public boolean tryIncRef() {
    int count;
    do {
      count = refCount.get();
      if (count == 0) return false; // already fully released; must not be resurrected
    } while (!refCount.compareAndSet(count, count + 1));
    return true;
  }

  @Override
  public void decRef() {
    int count = refCount.decrementAndGet();
    if (count == 0) {
      releaseDeviceData();
    } else if (count < 0) {
      throw new IllegalStateException("FilterBitsetHandle decRef() called without a matching ref");
    }
  }

  /**
   * Returns the shared device allocation for this filter, uploading on first call (lazy,
   * thread-safe via double-checked locking). The allocation is owned by {@link #filterResources()},
   * not the caller's resources.
   */
  DeviceData getOrUpload() {
    // Callers must hold a reference (tryIncRef) across this call and the subsequent device use, so
    // the allocation cannot be released underneath them; a zero count here means that contract was
    // violated.
    if (refCount.get() <= 0) {
      throw new IllegalStateException("FilterBitsetHandle has been released");
    }
    DeviceData data = sharedDeviceData;
    if (data != null) return data;
    synchronized (uploadLock) {
      data = sharedDeviceData;
      if (data != null) return data;
      data = upload();
      sharedDeviceData = data; // volatile write: happens-before all subsequent reads
    }
    return data;
  }

  private DeviceData upload() {
    long combinedBitsetBytes = (long) combinedLongs.length * Long.BYTES;
    // Serialize uploads: they share one resources (single stream/host-buffer) and are rare (once per
    // distinct filter, on cache miss). The free path uses the captured resources handle directly and
    // is RMM-thread-safe, so it needs no lock.
    synchronized (FILTER_RESOURCES_LOCK) {
      try (var access = filterResources().access()) {
        long cuvsRes = access.handle();
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
    }
  }

  /** Releases the initial reference held since construction. Idempotent. */
  @Override
  public void close() {
    if (closed.compareAndSet(false, true)) {
      decRef();
    }
  }

  /** Frees the shared device allocation. Called once, when the last reference is dropped. */
  private void releaseDeviceData() {
    DeviceData data;
    synchronized (uploadLock) {
      data = sharedDeviceData;
      sharedDeviceData = null;
    }
    if (data != null) data.close();
  }
}

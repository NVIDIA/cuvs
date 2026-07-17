/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import com.nvidia.cuvs.spi.CuVSProvider;

/**
 * Holds a precomputed multi-partition filter bitset and manages its device-memory lifecycle.
 *
 * <p>The packed {@code long[]} host arrays are immutable after construction. A single shared device
 * allocation is uploaded lazily on first use and reused thereafter.
 *
 * <h2>Lifecycle</h2>
 *
 * <p>The handle is reference-counted. Construction grants one initial reference, held by the owner
 * (typically a host-level cache), which is released by {@link #close()}. A thread that uses the
 * handle concurrently — e.g. while it may be evicted and closed by another thread — must guard the
 * use with {@link #tryIncRef()} / {@link #decRef()}. The shared device allocation is released only
 * when the last reference is dropped, so a concurrent {@link #close()} cannot free memory that is
 * still in use.
 *
 * @since 25.10
 */
public interface FilterBitsetHandle extends AutoCloseable {

  /**
   * Attempts to acquire a reference to this handle, preventing its device allocation from being
   * released until a matching {@link #decRef()}. Callers that pass the handle to a search (or
   * otherwise touch its device allocation) must hold a reference for the duration of that use.
   *
   * @return {@code true} if a reference was acquired; {@code false} if the handle has already been
   *     fully released, in which case no reference is acquired and it must not be used
   */
  boolean tryIncRef();

  /**
   * Releases a reference previously acquired via {@link #tryIncRef()}. When the last outstanding
   * reference is released, the shared device allocation is freed.
   *
   * @throws IllegalStateException if called without a matching {@link #tryIncRef()}
   */
  void decRef();

  /**
   * Creates a handle from the pre-packed combined bitset. Per-partition bit offsets are recomputed
   * inside cuVS from the index sizes, so only the concatenated (64-bit word-aligned) bitset words
   * are needed here.
   *
   * @param combinedLongs packed bitset words for all partitions concatenated (64-bit aligned)
   */
  static FilterBitsetHandle create(long[] combinedLongs) {
    return CuVSProvider.provider().newFilterBitsetHandle(combinedLongs);
  }

  /**
   * Releases the initial reference held since construction. Equivalent to a single {@link #decRef()}
   * of the owner's reference; the device allocation is freed once this and every reference acquired
   * via {@link #tryIncRef()} has been released. Idempotent — releasing the initial reference more
   * than once has no effect.
   */
  @Override
  void close();
}

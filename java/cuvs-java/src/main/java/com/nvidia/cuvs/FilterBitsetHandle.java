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
 * allocation is uploaded lazily on first use and reused thereafter; callers must {@link #close()}
 * the handle when it is evicted from their host-level cache.
 *
 * @since 25.10
 */
public interface FilterBitsetHandle extends AutoCloseable {

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

  /** Releases the shared device allocation associated with this handle. */
  @Override
  void close();
}

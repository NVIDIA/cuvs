/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal;

import java.util.OptionalLong;

/** Resolves the optional initial size of the filter bitset device pool. */
final class FilterBitsetPoolConfig {

  static final long DEFAULT_FILTER_POOL_BYTES = 4L << 20;
  private static final long RMM_ALIGNMENT_BYTES = 256L;
  private static final System.Logger LOG = System.getLogger(FilterBitsetPoolConfig.class.getName());

  private FilterBitsetPoolConfig() {}

  /**
   * Resolves a raw property value to an aligned initial pool size.
   *
   * <p>An absent value uses the default, zero explicitly disables the pool, and positive values
   * select an explicit initial size. Invalid, negative, and unalignable values warn and use the
   * default.
   */
  static OptionalLong resolvePoolBytes(String raw) {
    if (raw == null) {
      return OptionalLong.of(DEFAULT_FILTER_POOL_BYTES);
    }

    final long requestedBytes;
    try {
      requestedBytes = Long.parseLong(raw.trim());
    } catch (NumberFormatException e) {
      warnAndUseDefault(raw);
      return OptionalLong.of(DEFAULT_FILTER_POOL_BYTES);
    }

    if (requestedBytes == 0) {
      return OptionalLong.empty();
    }
    if (requestedBytes < 0 || requestedBytes > Long.MAX_VALUE - (RMM_ALIGNMENT_BYTES - 1)) {
      warnAndUseDefault(raw);
      return OptionalLong.of(DEFAULT_FILTER_POOL_BYTES);
    }

    long alignedBytes =
        (requestedBytes + (RMM_ALIGNMENT_BYTES - 1)) & ~(RMM_ALIGNMENT_BYTES - 1);
    return OptionalLong.of(alignedBytes);
  }

  private static void warnAndUseDefault(String raw) {
    LOG.log(
        System.Logger.Level.WARNING,
        "Invalid com.nvidia.cuvs.filterBitsetPoolSize value \""
            + raw
            + "\"; using the default "
            + DEFAULT_FILTER_POOL_BYTES
            + "-byte filter bitset device pool");
  }
}

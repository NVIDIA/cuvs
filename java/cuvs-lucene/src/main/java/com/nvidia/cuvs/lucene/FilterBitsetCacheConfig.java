/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

/**
 * Configuration for the filter-bitset cache used by multi-segment GPU searches.
 *
 * <p>This configuration is local to one vectors-format instance. It does not configure cuvs-java's
 * process-wide filter-bitset device pool.
 *
 * <p>The byte budget is a retention cap, not a preallocation. Packed host arrays and their
 * device-side mirrors are allocated as cache entries are populated. Actual host usage can exceed
 * the accounted payload bytes because of object overhead, temporary filter construction, and
 * in-flight handles.
 *
 * @param enabled whether filter bitsets are cached between searches
 * @param maxBytes maximum bytes cached by this format; must be positive when caching is enabled
 */
public record FilterBitsetCacheConfig(boolean enabled, long maxBytes) {

  /** Fixed 128 MiB cache budget used by SPI-created codecs and vector formats. */
  public static final long DEFAULT_MAX_BYTES = 128L * 1024 * 1024;

  /** Default configuration used by SPI-created codecs and vector formats. */
  public static final FilterBitsetCacheConfig DEFAULT =
      new FilterBitsetCacheConfig(true, DEFAULT_MAX_BYTES);

  public FilterBitsetCacheConfig {
    if (maxBytes < 0) {
      throw new IllegalArgumentException("maxBytes must be non-negative");
    }
    if (enabled && maxBytes == 0) {
      throw new IllegalArgumentException("maxBytes must be positive when caching is enabled");
    }
  }
}

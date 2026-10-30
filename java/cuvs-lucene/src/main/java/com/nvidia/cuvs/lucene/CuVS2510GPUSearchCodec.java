/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import org.apache.lucene.codecs.Codec;

/**
 * cuVS based codec for GPU based vector search that enables both - indexing and search on the GPU.
 * cuVS serialization formats are in experimental phase and hence backward compatibility cannot be guaranteed.
 *
 * <p>This codec wraps {@code Lucene101Codec}, the default codec of Lucene 10.2. On later Lucene
 * releases it can only read existing indexes; use {@link CuVSCodecs#gpuSearch} to write.
 *
 * @since 25.10
 */
public class CuVS2510GPUSearchCodec extends CuVSFilterCodec {

  private static final String NAME = "CuVS2510GPUSearchCodec";

  /**
   * Default constructor for {@link CuVS2510GPUSearchCodec}.
   */
  public CuVS2510GPUSearchCodec() {
    this(new GPUSearchParams.Builder().build(), FilterBitsetCacheConfig.DEFAULT);
  }

  /**
   * Initialize {@link CuVS2510GPUSearchCodec} with an instance of {@link GPUSearchParams}
   * having default parameter values.
   *
   * @param name the name of the codec
   * @param delegate the delegate codec
   */
  public CuVS2510GPUSearchCodec(String name, Codec delegate) {
    this(name, delegate, new GPUSearchParams.Builder().build(), FilterBitsetCacheConfig.DEFAULT);
  }

  /**
   * Initialize the codec with an instance of {@link GPUSearchParams} having either default
   * or overridden parameter values.
   *
   * @param params An instance of {@link GPUSearchParams}
   */
  public CuVS2510GPUSearchCodec(GPUSearchParams params) {
    this(params, FilterBitsetCacheConfig.DEFAULT);
  }

  /**
   * Initialize the codec with GPU search and filter-bitset-cache parameters.
   *
   * @param params GPU index and search parameters
   * @param filterCacheConfig filter-bitset-cache configuration
   */
  public CuVS2510GPUSearchCodec(GPUSearchParams params, FilterBitsetCacheConfig filterCacheConfig) {
    super(
        NAME,
        101,
        () -> LuceneCompat.lucene101Codec(),
        () -> new CuVS2510GPUVectorsFormat(params, filterCacheConfig));
  }

  /**
   * Initialize a named codec with explicit delegate, GPU search, and filter-cache parameters.
   *
   * @param name the name of the codec
   * @param delegate the delegate codec
   * @param params GPU index and search parameters
   * @param filterCacheConfig filter-bitset-cache configuration
   */
  public CuVS2510GPUSearchCodec(
      String name,
      Codec delegate,
      GPUSearchParams params,
      FilterBitsetCacheConfig filterCacheConfig) {
    super(name, delegate, () -> new CuVS2510GPUVectorsFormat(params, filterCacheConfig));
  }
}

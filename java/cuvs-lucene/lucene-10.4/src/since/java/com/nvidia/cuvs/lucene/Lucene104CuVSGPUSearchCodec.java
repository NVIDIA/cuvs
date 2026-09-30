/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

/**
 * A codec that both builds and searches CAGRA indexes on the GPU. cuVS serialization formats are in
 * experimental phase and hence backward compatibility cannot be guaranteed.
 *
 * <p>This codec wraps {@code Lucene104Codec}, the default codec of Lucene 10.4 and 10.5. Once a
 * later Lucene release replaces that default, this codec can only read existing indexes; use {@link
 * CuVSCodecs#gpuSearch} to write.
 *
 * @since 26.12
 */
public class Lucene104CuVSGPUSearchCodec extends CuVSFilterCodec {

  private static final String NAME = "Lucene104CuVSGPUSearchCodec";

  /** Creates the codec with default parameters. */
  public Lucene104CuVSGPUSearchCodec() {
    this(new GPUSearchParams.Builder().build(), FilterBitsetCacheConfig.DEFAULT);
  }

  /**
   * Creates the codec.
   *
   * @param params GPU index and search parameters
   * @param filterCacheConfig filter-bitset-cache configuration
   */
  public Lucene104CuVSGPUSearchCodec(
      GPUSearchParams params, FilterBitsetCacheConfig filterCacheConfig) {
    super(
        NAME,
        104,
        () -> LuceneCompat.lucene104Codec(),
        () -> new CuVS2510GPUVectorsFormat(params, filterCacheConfig));
  }
}

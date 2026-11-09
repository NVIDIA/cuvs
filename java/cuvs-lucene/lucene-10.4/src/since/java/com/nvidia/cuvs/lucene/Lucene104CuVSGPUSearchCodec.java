/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import org.apache.lucene.codecs.Codec;

/**
 * A codec that both builds and searches CAGRA indexes on the GPU. cuVS serialization formats are in
 * experimental phase and hence backward compatibility cannot be guaranteed.
 *
 * <p>This codec wraps {@code Lucene104Codec}, the default codec of Lucene 10.4. Once a later Lucene
 * release replaces that default, this codec can only read existing indexes; use {@link
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

  /**
   * Creates the codec with a delegate chosen by the caller, for applications that wrap a codec of
   * their own. Unlike the other constructors, this never makes the codec read-only: whether it can
   * write on a later Lucene release depends on the delegate, and on its vectors format, which a
   * later Lucene release may stop writing too.
   *
   * @param name the codec's name, which Lucene records in every segment the codec writes
   * @param delegate the codec to filter
   * @param params GPU index and search parameters
   * @param filterCacheConfig filter-bitset-cache configuration
   */
  public Lucene104CuVSGPUSearchCodec(
      String name,
      Codec delegate,
      GPUSearchParams params,
      FilterBitsetCacheConfig filterCacheConfig) {
    super(name, delegate, () -> new CuVS2510GPUVectorsFormat(params, filterCacheConfig));
  }
}

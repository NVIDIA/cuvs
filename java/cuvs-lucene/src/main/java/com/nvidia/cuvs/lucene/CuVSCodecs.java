/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;

/**
 * Creates the cuvs-lucene codecs and vectors formats that write indexes on the Lucene release this
 * artifact is built for.
 *
 * <p>Each cuvs-lucene codec wraps the default codec of one Lucene release, so the codec to write
 * with changes along with Lucene: {@code Lucene101AcceleratedHNSWCodec} on Lucene 10.2, {@code
 * Lucene103AcceleratedHNSWCodec} on 10.3, {@code Lucene104AcceleratedHNSWCodec} on 10.4 and later.
 * The older codecs stay available for reading existing indexes. Creating codecs through this class
 * instead of naming them keeps application code unchanged when it moves to another Lucene release
 * and the matching cuvs-lucene artifact.
 *
 * <p>Every method throws {@link IllegalStateException} if the Lucene release in use is not the one
 * this cuvs-lucene artifact is built for.
 *
 * @since 26.12
 */
public final class CuVSCodecs {

  private CuVSCodecs() {}

  /**
   * Checks that the Lucene release in use is the one this cuvs-lucene artifact is built for.
   *
   * <p>A cuvs-lucene artifact on the wrong Lucene release does not break Lucene: Lucene keeps
   * working, and so do indexes that do not use the cuvs-lucene codecs, while using one of these
   * codecs fails. Applications that would rather not start at all in that case can call this method
   * during startup.
   *
   * @throws IllegalStateException if the artifact is not built for the Lucene release in use; the
   *     message names the artifact to use instead
   */
  public static void checkLuceneVersion() {
    LuceneVersionGuard.ensureCompatible();
  }

  /**
   * Returns the codec that builds HNSW graphs on the GPU and searches them on the CPU, with default
   * parameters.
   *
   * @return the codec
   */
  public static Codec acceleratedHNSW() {
    return acceleratedHNSW(new AcceleratedHNSWParams.Builder().build());
  }

  /**
   * Returns the codec that builds HNSW graphs on the GPU and searches them on the CPU.
   *
   * @param params the index build parameters
   * @return the codec
   */
  public static Codec acceleratedHNSW(AcceleratedHNSWParams params) {
    LuceneVersionGuard.ensureCompatible();
    return LuceneCompat.acceleratedHNSWCodec(params);
  }

  /**
   * Returns the codec that builds HNSW graphs on the GPU over scalar-quantized vectors, with
   * default parameters.
   *
   * @return the codec
   */
  public static Codec acceleratedHNSWScalarQuantized() {
    return acceleratedHNSWScalarQuantized(new AcceleratedHNSWParams.Builder().build());
  }

  /**
   * Returns the codec that builds HNSW graphs on the GPU over scalar-quantized vectors.
   *
   * @param params the index build parameters
   * @return the codec
   */
  public static Codec acceleratedHNSWScalarQuantized(AcceleratedHNSWParams params) {
    LuceneVersionGuard.ensureCompatible();
    return LuceneCompat.acceleratedHNSWScalarQuantizedCodec(params);
  }

  /**
   * Returns the vectors format used by {@link
   * #acceleratedHNSWScalarQuantized(AcceleratedHNSWParams)}, for use with a per-field codec.
   *
   * @param params the index build parameters
   * @return the vectors format
   */
  public static KnnVectorsFormat acceleratedHNSWScalarQuantizedFormat(
      AcceleratedHNSWParams params) {
    LuceneVersionGuard.ensureCompatible();
    return LuceneCompat.acceleratedHNSWScalarQuantizedFormat(params);
  }

  /**
   * Returns the codec that builds HNSW graphs on the GPU over binary-quantized vectors, with
   * default parameters.
   *
   * @return the codec
   */
  public static Codec acceleratedHNSWBinaryQuantized() {
    return acceleratedHNSWBinaryQuantized(new AcceleratedHNSWParams.Builder().build());
  }

  /**
   * Returns the codec that builds HNSW graphs on the GPU over binary-quantized vectors.
   *
   * @param params the index build parameters
   * @return the codec
   */
  public static Codec acceleratedHNSWBinaryQuantized(AcceleratedHNSWParams params) {
    LuceneVersionGuard.ensureCompatible();
    return LuceneCompat.acceleratedHNSWBinaryQuantizedCodec(params);
  }

  /**
   * Returns the codec that both builds and searches CAGRA indexes on the GPU, with default
   * parameters.
   *
   * @return the codec
   */
  public static Codec gpuSearch() {
    return gpuSearch(new GPUSearchParams.Builder().build());
  }

  /**
   * Returns the codec that both builds and searches CAGRA indexes on the GPU, with the default
   * filter-bitset-cache configuration.
   *
   * @param params the GPU index and search parameters
   * @return the codec
   */
  public static Codec gpuSearch(GPUSearchParams params) {
    return gpuSearch(params, FilterBitsetCacheConfig.DEFAULT);
  }

  /**
   * Returns the codec that both builds and searches CAGRA indexes on the GPU.
   *
   * @param params the GPU index and search parameters
   * @param filterCacheConfig the filter-bitset-cache configuration
   * @return the codec
   */
  public static Codec gpuSearch(GPUSearchParams params, FilterBitsetCacheConfig filterCacheConfig) {
    LuceneVersionGuard.ensureCompatible();
    return LuceneCompat.gpuSearchCodec(params, filterCacheConfig);
  }
}

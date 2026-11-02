/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.lucene101.Lucene101Codec;
import org.apache.lucene.codecs.lucene99.Lucene99ScalarQuantizedVectorsFormat;

/**
 * The Lucene calls that differ between the Lucene releases cuvs-lucene supports, for Lucene 10.2.
 *
 * <p>Every lucene-X.Y module has its own copy of this class, compiled against that release. The
 * shared sources go through it instead of calling those parts of Lucene directly.
 */
final class LuceneCompat {

  // The Lucene release this module is built for. Both must stay compile-time constants: javac
  // copies them into the classes that read them, so LuceneVersionGuard can check the Lucene version
  // without loading this class, which does not link against other Lucene releases.
  static final int LUCENE_MAJOR = 10;
  static final int LUCENE_MINOR = 2;

  private LuceneCompat() {}

  /** Returns Lucene 10.2's {@code Lucene101Codec}. */
  static Codec lucene101Codec() {
    return new Lucene101Codec();
  }

  /** Returns the flat format that stores the vectors of the scalar-quantized codecs. */
  static FlatVectorsFormat lucene99ScalarQuantizedFlatFormat() {
    return new Lucene99ScalarQuantizedVectorsFormat();
  }

  // The codecs for this Lucene release, returned by CuVSCodecs.

  static Codec acceleratedHNSWCodec(AcceleratedHNSWParams params) {
    return new Lucene101AcceleratedHNSWCodec(params);
  }

  static KnnVectorsFormat acceleratedHNSWFormat(AcceleratedHNSWParams params) {
    return new Lucene99AcceleratedHNSWVectorsFormat(params);
  }

  static Codec acceleratedHNSWScalarQuantizedCodec(AcceleratedHNSWParams params) {
    return new LuceneAcceleratedHNSWScalarQuantizedCodec(params);
  }

  static KnnVectorsFormat acceleratedHNSWScalarQuantizedFormat(AcceleratedHNSWParams params) {
    return new LuceneAcceleratedHNSWScalarQuantizedVectorsFormat(params);
  }

  static Codec acceleratedHNSWBinaryQuantizedCodec(AcceleratedHNSWParams params) {
    return new LuceneAcceleratedHNSWBinaryQuantizedCodec(params);
  }

  static KnnVectorsFormat acceleratedHNSWBinaryQuantizedFormat(AcceleratedHNSWParams params) {
    return new LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat(params);
  }

  static Codec gpuSearchCodec(GPUSearchParams params, FilterBitsetCacheConfig filterCacheConfig) {
    return new CuVS2510GPUSearchCodec(params, filterCacheConfig);
  }

  static KnnVectorsFormat gpuSearchFormat(
      GPUSearchParams params, FilterBitsetCacheConfig filterCacheConfig) {
    return new CuVS2510GPUVectorsFormat(params, filterCacheConfig);
  }
}

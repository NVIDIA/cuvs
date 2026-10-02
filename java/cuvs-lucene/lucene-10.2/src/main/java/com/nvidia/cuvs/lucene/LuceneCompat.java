/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.lucene101.Lucene101Codec;
import org.apache.lucene.codecs.lucene99.Lucene99HnswScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99ScalarQuantizedVectorsFormat;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.ReadAdvice;

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

  // The codec generation that writes on this Lucene release: the version number of the Lucene
  // codec that codecs of this generation wrap (Lucene101Codec). Codecs of other generations are
  // read-only. Also a compile-time constant, for the same reason.
  static final int CURRENT_CODEC_GENERATION = 101;

  private LuceneCompat() {}

  /** Returns {@code context} with a hint that the file will be read sequentially. */
  static IOContext sequentialReadContext(IOContext context) {
    return context.withReadAdvice(ReadAdvice.SEQUENTIAL);
  }

  /** Returns Lucene 10.2's {@code Lucene101Codec}. */
  static Codec lucene101Codec() {
    return new Lucene101Codec();
  }

  /** Returns the flat format that stores the vectors of the Lucene 10.2 and 10.3 SQ codecs. */
  static FlatVectorsFormat lucene99ScalarQuantizedFlatFormat() {
    return new Lucene99ScalarQuantizedVectorsFormat();
  }

  /**
   * Whether this Lucene release can write {@link #lucene99ScalarQuantizedFlatFormat()}. Lucene 10.2
   * still writes it.
   */
  static boolean canWriteLucene99ScalarQuantized() {
    return true;
  }

  /**
   * Returns Lucene's HNSW format over {@link #lucene99ScalarQuantizedFlatFormat()}, which builds
   * the graph on the CPU.
   */
  static KnnVectorsFormat lucene99HnswScalarQuantizedFormat(AcceleratedHNSWParams params) {
    int numMergeWorkers = params.getNumMergeWorkers();
    return new Lucene99HnswScalarQuantizedVectorsFormat(
        params.getMaxConn(),
        params.getBeamWidth(),
        numMergeWorkers,
        7,
        false,
        null,
        numMergeWorkers > 1 ? params.getMergeExec() : null);
  }

  // The codecs that write on this Lucene release, returned by CuVSCodecs.

  static Codec acceleratedHNSWCodec(AcceleratedHNSWParams params) {
    return new Lucene101AcceleratedHNSWCodec(params);
  }

  static Codec acceleratedHNSWScalarQuantizedCodec(AcceleratedHNSWParams params) {
    return new LuceneAcceleratedHNSWScalarQuantizedCodec(params);
  }

  /** Returns the name of the format {@link #acceleratedHNSWScalarQuantizedFormat} creates. */
  static String acceleratedHNSWScalarQuantizedFormatName() {
    return LuceneAcceleratedHNSWScalarQuantizedVectorsFormat.NAME;
  }

  static KnnVectorsFormat acceleratedHNSWScalarQuantizedFormat(AcceleratedHNSWParams params) {
    return new LuceneAcceleratedHNSWScalarQuantizedVectorsFormat(params);
  }

  static Codec acceleratedHNSWBinaryQuantizedCodec(AcceleratedHNSWParams params) {
    return new LuceneAcceleratedHNSWBinaryQuantizedCodec(params);
  }

  static Codec gpuSearchCodec(GPUSearchParams params, FilterBitsetCacheConfig filterCacheConfig) {
    return new CuVS2510GPUSearchCodec(params, filterCacheConfig);
  }
}

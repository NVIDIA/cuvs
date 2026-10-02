/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import org.apache.lucene.backward_codecs.lucene101.Lucene101Codec;
import org.apache.lucene.backward_codecs.lucene103.Lucene103Codec;
import org.apache.lucene.backward_codecs.lucene99.Lucene99ScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.lucene104.Lucene104HnswScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat.ScalarEncoding;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.IOContext;

/**
 * The Lucene calls that differ between the Lucene releases cuvs-lucene supports, for Lucene 10.4.
 *
 * <p>Every lucene-X.Y module has its own copy of this class, compiled against that release. The
 * shared sources go through it instead of calling those parts of Lucene directly.
 */
final class LuceneCompat {

  // The Lucene release this module is built for. Both must stay compile-time constants: javac
  // copies them into the classes that read them, so LuceneVersionGuard can check the Lucene version
  // without loading this class, which does not link against other Lucene releases.
  static final int LUCENE_MAJOR = 10;
  static final int LUCENE_MINOR = 4;

  // The codec generation that writes on this Lucene release: the version number of the Lucene
  // codec that codecs of this generation wrap (Lucene104Codec). Codecs of other generations are
  // read-only. Also a compile-time constant, for the same reason.
  static final int CURRENT_CODEC_GENERATION = 104;

  private LuceneCompat() {}

  /** Returns {@code context} with a hint that the file will be read sequentially. */
  static IOContext sequentialReadContext(IOContext context) {
    return context.withHints(DataAccessHint.SEQUENTIAL);
  }

  /** Returns Lucene 10.2's {@code Lucene101Codec}, which Lucene 10.4 can only read. */
  static Codec lucene101Codec() {
    return new Lucene101Codec();
  }

  /** Returns Lucene 10.3's {@code Lucene103Codec}, which Lucene 10.4 can only read. */
  static Codec lucene103Codec() {
    return new Lucene103Codec();
  }

  /** Returns Lucene 10.4's {@code Lucene104Codec}. */
  static Codec lucene104Codec() {
    return new Lucene104Codec();
  }

  /**
   * Returns the flat format that stores the vectors of the Lucene 10.2 and 10.3 SQ codecs, which
   * Lucene 10.4 can only read.
   */
  static FlatVectorsFormat lucene99ScalarQuantizedFlatFormat() {
    return new Lucene99ScalarQuantizedVectorsFormat();
  }

  /**
   * Whether this Lucene release can write {@link #lucene99ScalarQuantizedFlatFormat()}. Lucene 10.4
   * moved it to its backward codecs, which only read.
   */
  static boolean canWriteLucene99ScalarQuantized() {
    return false;
  }

  /** Lucene 10.4 cannot write {@link #lucene99ScalarQuantizedFlatFormat()}, so this throws. */
  static KnnVectorsFormat lucene99HnswScalarQuantizedFormat(AcceleratedHNSWParams params) {
    throw new UnsupportedOperationException("Old codecs may only be used for reading");
  }

  /** Returns the flat format that stores the vectors of the SQ codecs since Lucene 10.4. */
  static FlatVectorsFormat lucene104ScalarQuantizedFlatFormat() {
    return new Lucene104ScalarQuantizedVectorsFormat(ScalarEncoding.SEVEN_BIT);
  }

  /**
   * Returns Lucene's HNSW format over {@link #lucene104ScalarQuantizedFlatFormat()}, which builds
   * the graph on the CPU.
   */
  static KnnVectorsFormat lucene104HnswScalarQuantizedFormat(AcceleratedHNSWParams params) {
    int numMergeWorkers = params.getNumMergeWorkers();
    return new Lucene104HnswScalarQuantizedVectorsFormat(
        ScalarEncoding.SEVEN_BIT,
        params.getMaxConn(),
        params.getBeamWidth(),
        numMergeWorkers,
        numMergeWorkers > 1 ? params.getMergeExec() : null);
  }

  // The codecs that write on this Lucene release, returned by CuVSCodecs.

  static Codec acceleratedHNSWCodec(AcceleratedHNSWParams params) {
    return new Lucene104AcceleratedHNSWCodec(params);
  }

  static Codec acceleratedHNSWScalarQuantizedCodec(AcceleratedHNSWParams params) {
    return new Lucene104AcceleratedHNSWScalarQuantizedCodec(params);
  }

  static KnnVectorsFormat acceleratedHNSWScalarQuantizedFormat(AcceleratedHNSWParams params) {
    return new Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat(params);
  }

  static Codec acceleratedHNSWBinaryQuantizedCodec(AcceleratedHNSWParams params) {
    return new Lucene104AcceleratedHNSWBinaryQuantizedCodec(params);
  }

  static Codec gpuSearchCodec(GPUSearchParams params, FilterBitsetCacheConfig filterCacheConfig) {
    return new Lucene104CuVSGPUSearchCodec(params, filterCacheConfig);
  }
}

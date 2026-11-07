/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import org.apache.lucene.backward_codecs.lucene101.Lucene101Codec;
import org.apache.lucene.backward_codecs.lucene103.Lucene103Codec;
import org.apache.lucene.backward_codecs.lucene99.Lucene99ScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.FileDataHint;
import org.apache.lucene.store.FileTypeHint;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;

/**
 * The Lucene calls that differ between the Lucene releases cuvs-lucene supports, for Lucene 10.6.
 *
 * <p>Every lucene-X.Y module has its own copy of this class, compiled against that release. The
 * shared sources go through it instead of calling those parts of Lucene directly.
 */
final class LuceneCompat {

  // The Lucene release this module is built for. Both must stay compile-time constants: javac
  // copies them into the classes that read them, so LuceneVersionGuard can check the Lucene version
  // without loading this class, which does not link against other Lucene releases.
  static final int LUCENE_MAJOR = 10;
  static final int LUCENE_MINOR = 6;

  // The codec generation that writes on this Lucene release: the version number of the Lucene
  // codec that codecs of this generation wrap (Lucene104Codec). Codecs of other generations are
  // read-only. Also a compile-time constant, for the same reason.
  static final int CURRENT_CODEC_GENERATION = 104;

  // The registered name of the scalar-quantized format that writes on this Lucene release, the one
  // CuVSCodecs.acceleratedHNSWScalarQuantizedFormat returns. A constant, so that the formats can
  // check it without creating one.
  static final String SCALAR_QUANTIZED_FORMAT_NAME =
      "Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat";

  private LuceneCompat() {}

  /** Returns {@code context} with a hint that the file will be read sequentially. */
  static IOContext sequentialReadContext(IOContext context) {
    return context.withHints(DataAccessHint.SEQUENTIAL);
  }

  /**
   * Returns {@code state} for opening the flat vectors under an HNSW graph, which searches read at
   * random. Since Lucene 10.6 the flat format no longer says so itself, and leaves it to the format
   * that wraps it, as Lucene's HNSW formats do.
   */
  static SegmentReadState flatReadState(SegmentReadState state) {
    return state.withHints(FileTypeHint.DATA, FileDataHint.KNN_VECTORS, DataAccessHint.RANDOM);
  }

  /**
   * Returns Lucene's HNSW writer, which builds the graph on the CPU, over {@code flatWriter}, which
   * {@code flatFormat} created. Since Lucene 10.5, it reads the vectors of a merged segment back with
   * {@code flatFormat} to build its graph.
   */
  static KnnVectorsWriter hnswVectorsWriter(
      SegmentWriteState state,
      int maxConn,
      int beamWidth,
      FlatVectorsFormat flatFormat,
      FlatVectorsWriter flatWriter,
      int numMergeWorkers,
      TaskExecutor mergeExec)
      throws IOException {
    return new Lucene99HnswVectorsWriter(
        state, maxConn, beamWidth, flatFormat, flatWriter, numMergeWorkers, mergeExec);
  }

  /** Returns Lucene 10.2's {@code Lucene101Codec}, which Lucene 10.6 can only read. */
  static Codec lucene101Codec() {
    return new Lucene101Codec();
  }

  /** Returns Lucene 10.3's {@code Lucene103Codec}, which Lucene 10.6 can only read. */
  static Codec lucene103Codec() {
    return new Lucene103Codec();
  }

  /** Returns {@code Lucene104Codec}, the default codec of Lucene 10.4, 10.5 and 10.6. */
  static Codec lucene104Codec() {
    return new Lucene104Codec();
  }

  /**
   * Returns the flat format that stores the vectors of the scalar-quantized codecs up to Lucene
   * 10.3, which Lucene 10.6 can only read.
   */
  static FlatVectorsFormat lucene99ScalarQuantizedFlatFormat() {
    return new Lucene99ScalarQuantizedVectorsFormat();
  }

  /**
   * Returns the flat format that stores the vectors of the scalar-quantized codecs since Lucene
   * 10.4, with one byte per dimension: Lucene's default encoding, {@code UNSIGNED_BYTE}, which is
   * the same size as {@code SEVEN_BIT} but more accurate. The format records the encoding of each
   * field, so a later change here keeps existing segments readable.
   */
  static FlatVectorsFormat lucene104ScalarQuantizedFlatFormat() {
    return new Lucene104ScalarQuantizedVectorsFormat(ScalarEncoding.UNSIGNED_BYTE);
  }

  // The codecs for this Lucene release, returned by CuVSCodecs.

  static Codec acceleratedHNSWCodec(AcceleratedHNSWParams params) {
    return new Lucene104AcceleratedHNSWCodec(params);
  }

  static KnnVectorsFormat acceleratedHNSWFormat(AcceleratedHNSWParams params) {
    return new Lucene99AcceleratedHNSWVectorsFormat(params);
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

  static KnnVectorsFormat acceleratedHNSWBinaryQuantizedFormat(AcceleratedHNSWParams params) {
    return new LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat(params);
  }

  static Codec gpuSearchCodec(GPUSearchParams params, FilterBitsetCacheConfig filterCacheConfig) {
    return new Lucene104CuVSGPUSearchCodec(params, filterCacheConfig);
  }

  static KnnVectorsFormat gpuSearchFormat(
      GPUSearchParams params, FilterBitsetCacheConfig filterCacheConfig) {
    return new CuVS2510GPUVectorsFormat(params, filterCacheConfig);
  }
}

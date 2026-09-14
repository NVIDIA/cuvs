/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;

import com.nvidia.cuvs.LibraryException;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.search.TaskExecutor;

/**
 * cuVS based KnnVectorsFormat for indexing on GPU and searching on the CPU.
 *
 * @since 25.10
 */
public class Lucene99AcceleratedHNSWVectorsFormat extends KnnVectorsFormat {

  private static final Logger log =
      Logger.getLogger(Lucene99AcceleratedHNSWVectorsFormat.class.getName());
  private static final FlatVectorsFormat FLAT_VECTORS_FORMAT;
  private static final int MAX_DIMENSIONS = 4096;
  private final AcceleratedHNSWParams acceleratedHNSWParams;
  private final int numInputVectors;

  static final String HNSW_META_CODEC_NAME = "Lucene99HnswVectorsFormatMeta";
  static final String HNSW_META_CODEC_EXT = "vem";
  static final String HNSW_INDEX_CODEC_NAME = "Lucene99HnswVectorsFormatIndex";
  static final String HNSW_INDEX_EXT = "vex";
  static final LuceneProvider LUCENE_PROVIDER;

  static {
    try {
      LUCENE_PROVIDER = LuceneProvider.getInstance("99");
      FLAT_VECTORS_FORMAT =
          LUCENE_PROVIDER.getLuceneFlatVectorsFormatInstance(DefaultFlatVectorScorer.INSTANCE);
    } catch (Exception e) {
      throw new ExceptionInInitializerError(e.getMessage());
    }
  }

  /**
   * Initializes {@link Lucene99AcceleratedHNSWVectorsFormat} with an instance
   * of {@link AcceleratedHNSWParams} with default parameter values.
   *
   * @throws LibraryException if the native library fails to load
   */
  public Lucene99AcceleratedHNSWVectorsFormat() {
    this(new AcceleratedHNSWParams.Builder().build());
  }

  /**
   * Initializes {@link Lucene99AcceleratedHNSWVectorsFormat} with an instance
   * of {@link AcceleratedHNSWParams}.
   *
   * @param acceleratedHNSWParams An instance of {@link AcceleratedHNSWParams}
   */
  public Lucene99AcceleratedHNSWVectorsFormat(AcceleratedHNSWParams acceleratedHNSWParams) {
    this(acceleratedHNSWParams, 0);
  }

  /**
   * Initializes {@link Lucene99AcceleratedHNSWVectorsFormat} for the native flat-buffered build
   * path (see {@link NativeFlatBufferedHNSWVectorsWriter}). Package-private: only {@link
   * CagraHnswBulkIndexWriter} (via {@link Lucene101AcceleratedHNSWCodec}) can guarantee the
   * invariants that native flat buffering requires, so this overload is not part of the public
   * API.
   *
   * @param acceleratedHNSWParams An instance of {@link AcceleratedHNSWParams}
   * @param numInputVectors the exact number of vectors to be indexed, used to pre-size the native
   *     flat buffer (0 = disabled, the default heap-buffered path)
   */
  Lucene99AcceleratedHNSWVectorsFormat(
      AcceleratedHNSWParams acceleratedHNSWParams, int numInputVectors) {
    super("Lucene99AcceleratedHNSWVectorsFormat");
    this.acceleratedHNSWParams = acceleratedHNSWParams;
    this.numInputVectors = numInputVectors;
  }

  /**
   * Returns a KnnVectorsWriter to write the vectors to the index.
   */
  @Override
  public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    boolean nativeMode = isSupported() && numInputVectors > 0;
    if (isSupported()) {
      if (nativeMode) {
        log.log(Level.FINE, "cuVS is supported so using the NativeFlatBufferedHNSWVectorsWriter");
        // In hint mode the accelerated writer owns the flat .vec/.vemf files, so the Lucene flat
        // writer must not be created (it would open the same outputs).
        return new NativeFlatBufferedHNSWVectorsWriter(
            state, acceleratedHNSWParams, numInputVectors);
      }
      log.log(Level.FINE, "cuVS is supported so using the Lucene99AcceleratedHNSWVectorsWriter");
      var flatWriter = FLAT_VECTORS_FORMAT.fieldsWriter(state);
      return new Lucene99AcceleratedHNSWVectorsWriter(state, acceleratedHNSWParams, flatWriter);
    } else {
      log.log(
          Level.WARNING,
          "GPU based indexing not supported, falling back to using the Lucene99HnswVectorsWriter");
      var flatWriter = FLAT_VECTORS_FORMAT.fieldsWriter(state);
      try {
        return LUCENE_PROVIDER.getLuceneHnswVectorsWriterInstance(
            state,
            acceleratedHNSWParams.getMaxConn(),
            acceleratedHNSWParams.getBeamWidth(),
            flatWriter,
            acceleratedHNSWParams.getNumMergeWorkers(),
            new TaskExecutor(acceleratedHNSWParams.getMergeExec()));
      } catch (Exception e) {
        throw Utils.handleThrowable(e);
      }
    }
  }

  /**
   * Returns a KnnVectorsReader to read the vectors from the index.
   */
  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    try {
      return LUCENE_PROVIDER.getLuceneHnswVectorsReaderInstance(
          state, FLAT_VECTORS_FORMAT.fieldsReader(state));
    } catch (Exception e) {
      throw Utils.handleThrowable(e);
    }
  }

  /**
   * Returns the maximum number of vector dimensions supported by this codec for the given field name.
   */
  @Override
  public int getMaxDimensions(String fieldName) {
    return MAX_DIMENSIONS;
  }
}

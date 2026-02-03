/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

/**
 * cuVS based KnnVectorsFormat for indexing on GPU and searching on the CPU.
 *
 * @since 25.10
 */
public class Lucene99AcceleratedHNSWVectorsFormat extends KnnVectorsFormat {

  private static final Logger log =
      Logger.getLogger(Lucene99AcceleratedHNSWVectorsFormat.class.getName());
  private static final FlatVectorsFormat FLAT_VECTORS_FORMAT;
  private static final Integer NUM_MERGE_WORKERS;
  private static final int maxDimensions = 4096;
  private AcceleratedHNSWParams acceleratedHNSWParams;

  static final String HNSW_META_CODEC_NAME = "Lucene99HnswVectorsFormatMeta";
  static final String HNSW_META_CODEC_EXT = "vem";
  static final String HNSW_INDEX_CODEC_NAME = "Lucene99HnswVectorsFormatIndex";
  static final String HNSW_INDEX_EXT = "vex";
  static final LuceneProvider LUCENE_PROVIDER;

  static {
    try {
      LUCENE_PROVIDER = LuceneProvider.getInstance("99");
      NUM_MERGE_WORKERS = LUCENE_PROVIDER.getStaticIntParam("DEFAULT_BEAM_WIDTH");
      FLAT_VECTORS_FORMAT =
          LUCENE_PROVIDER.getLuceneFlatVectorsFormatInstance(DefaultFlatVectorScorer.INSTANCE);
    } catch (Exception e) {
      throw new ExceptionInInitializerError(e.getMessage());
    }
  }

  /**
   * Initializes {@link Lucene99AcceleratedHNSWVectorsFormat} with default parameter values.
   *
   * @throws LibraryException if the native library fails to load
   */
  public Lucene99AcceleratedHNSWVectorsFormat() {
    this(new AcceleratedHNSWParams.Builder().build());
  }

  /**
   * Initializes {@link Lucene99AcceleratedHNSWVectorsFormat} with the given threads, graph degree, etc.
   *
   * @param acceleratedHNSWParams An instance of {@link AcceleratedHNSWParams}
   */
  public Lucene99AcceleratedHNSWVectorsFormat(AcceleratedHNSWParams acceleratedHNSWParams) {
    super("Lucene99AcceleratedHNSWVectorsFormat");
    this.acceleratedHNSWParams = acceleratedHNSWParams;
  }

  /**
   * Returns a KnnVectorsWriter to write the vectors to the index.
   */
  @Override
  public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    var flatWriter = FLAT_VECTORS_FORMAT.fieldsWriter(state);
    if (isSupported()) {
      log.log(Level.FINE, "cuVS is supported so using the Lucene99AcceleratedHNSWVectorsWriter");
      return new Lucene99AcceleratedHNSWVectorsWriter(state, acceleratedHNSWParams, flatWriter);
    } else {
      log.log(
          Level.WARNING,
          "GPU based indexing not supported, falling back to using the Lucene99HnswVectorsWriter");
      // TODO: Make num merge workers configurable.
      try {
        return LUCENE_PROVIDER.getLuceneHnswVectorsWriterInstance(
            state,
            acceleratedHNSWParams.getMaxConn(),
            acceleratedHNSWParams.getBeamWidth(),
            flatWriter,
            NUM_MERGE_WORKERS,
            null);
      } catch (Exception e) {
        // maybe there is a better suited option to throwing RuntimeException? Need to explore.
        throw new RuntimeException(e.getMessage());
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
      // maybe there is a better suited option to throwing RuntimeException? Need to explore.
      throw new RuntimeException(e.getMessage());
    }
  }

  /**
   * Returns the maximum number of vector dimensions supported by this codec for the given field name.
   */
  @Override
  public int getMaxDimensions(String fieldName) {
    return maxDimensions;
  }
}

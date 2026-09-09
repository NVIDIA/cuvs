/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;

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
  private static volatile FlatVectorsFormat cachedFlatVectorsFormat;
  private static final int MAX_DIMENSIONS = 4096;
  private final AcceleratedHNSWParams acceleratedHNSWParams;

  static final String HNSW_META_CODEC_NAME = "Lucene99HnswVectorsFormatMeta";
  static final String HNSW_META_CODEC_EXT = "vem";
  static final String HNSW_INDEX_CODEC_NAME = "Lucene99HnswVectorsFormatIndex";
  static final String HNSW_INDEX_EXT = "vex";

  private static LuceneProvider getLucene99Provider() throws IOException {
    try {
      return LuceneProvider.getInstance(LuceneProvider.LUCENE_99_FORMAT_VERSION);
    } catch (ClassNotFoundException e) {
      throw new IOException("Lucene99 vector formats are not available in this runtime", e);
    }
  }

  private static FlatVectorsFormat getOrCreateFlatVectorsFormat() throws IOException {
    FlatVectorsFormat format = cachedFlatVectorsFormat;
    if (format == null) {
      synchronized (Lucene99AcceleratedHNSWVectorsFormat.class) {
        format = cachedFlatVectorsFormat;
        if (format == null) {
          try {
            format =
                getLucene99Provider()
                    .getLuceneFlatVectorsFormatInstance(DefaultFlatVectorScorer.INSTANCE);
            cachedFlatVectorsFormat = format;
          } catch (Exception e) {
            throw Utils.handleThrowable(e);
          }
        }
      }
    }
    return format;
  }

  /**
   * Initializes {@link Lucene99AcceleratedHNSWVectorsFormat} with an instance
   * of {@link AcceleratedHNSWParams} with default parameter values.
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
    super("Lucene99AcceleratedHNSWVectorsFormat");
    this.acceleratedHNSWParams = acceleratedHNSWParams;
  }

  /**
   * Returns a KnnVectorsWriter to write the vectors to the index.
   */
  @Override
  public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    var flatWriter = getOrCreateFlatVectorsFormat().fieldsWriter(state);
    if (isSupported()) {
      log.log(Level.FINE, "cuVS is supported so using the Lucene99AcceleratedHNSWVectorsWriter");
      return new Lucene99AcceleratedHNSWVectorsWriter(state, acceleratedHNSWParams, flatWriter);
    } else {
      log.log(
          Level.WARNING,
          "GPU based indexing not supported, falling back to using the Lucene99HnswVectorsWriter");
      try {
        return getLucene99Provider()
            .getLuceneHnswVectorsWriterInstance(
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
      return getLucene99Provider()
          .getLuceneHnswVectorsReaderInstance(
              state, getOrCreateFlatVectorsFormat().fieldsReader(state));
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

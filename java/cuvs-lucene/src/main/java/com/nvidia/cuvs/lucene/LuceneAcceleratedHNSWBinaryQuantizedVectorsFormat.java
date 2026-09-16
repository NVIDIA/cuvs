/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * cuVS based Binary Quantized KnnVectorsFormat for indexing on GPU and searching on the CPU.
 *
 * @since 26.02
 */
public class LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat extends KnnVectorsFormat {

  private static final Logger log =
      Logger.getLogger(LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.class.getName());
  private static volatile FlatVectorsFormat cachedFlatVectorsFormat;
  private static final int MAX_DIMENSIONS = 4096;

  private final AcceleratedHNSWParams acceleratedHNSWParams;

  private static LuceneProvider getLucene99Provider() throws IOException {
    try {
      return LuceneProvider.getInstance("99");
    } catch (ClassNotFoundException e) {
      throw new IOException("Lucene99 vector formats are not available in this runtime", e);
    }
  }

  private static LuceneProvider getLucene102Provider() throws IOException {
    try {
      return LuceneProvider.getInstance("102");
    } catch (ClassNotFoundException e) {
      throw new IOException("Lucene102 vector formats are not available in this runtime", e);
    }
  }

  private static FlatVectorsFormat getOrCreateFlatVectorsFormat() throws IOException {
    FlatVectorsFormat format = cachedFlatVectorsFormat;
    if (format == null) {
      synchronized (LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.class) {
        format = cachedFlatVectorsFormat;
        if (format == null) {
          try {
            // Keep the released Lucene99 flat-vector layout. Changing it requires an explicit
            // persisted-format migration so existing segments remain readable.
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
   * Initializes {@link LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat} with default values.
   */
  public LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat() {
    this(new AcceleratedHNSWParams.Builder().build());
  }

  /**
   * Initializes {@link LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat} with the given threads, graph degree, etc.
   *
   * @param acceleratedHNSWParams An instance of {@link AcceleratedHNSWParams}
   */
  public LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat(
      AcceleratedHNSWParams acceleratedHNSWParams) {
    super("Lucene99AcceleratedHNSWBinaryQuantizedVectorsFormat");
    this.acceleratedHNSWParams = acceleratedHNSWParams;
  }

  /**
   * Returns a KnnVectorsWriter to write the binary quantized vectors to the index.
   */
  @Override
  public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    var flatWriter = getOrCreateFlatVectorsFormat().fieldsWriter(state);
    if (isSupported()) {
      log.log(
          Level.FINE,
          "cuVS is supported so using the Lucene99AcceleratedHNSWBinaryQuantizedVectorsWriter");
      return new LuceneAcceleratedHNSWBinaryQuantizedVectorsWriter(
          state, acceleratedHNSWParams, flatWriter);
    } else {
      try {
        // Fallback to Lucene's Lucene102HnswBinaryQuantizedVectorsFormat format
        log.log(
            Level.WARNING,
            "GPU based indexing not supported, falling back to using the"
                + " Lucene102HnswBinaryQuantizedVectorsFormat");
        KnnVectorsFormat fallbackFormat =
            getLucene102Provider()
                .getLuceneHnswBinaryQuantizedVectorsFormatInstance(
                    acceleratedHNSWParams.getMaxConn(), acceleratedHNSWParams.getBeamWidth());
        return fallbackFormat.fieldsWriter(state);
      } catch (Exception e) {
        throw Utils.handleThrowable(e);
      }
    }
  }

  /**
   * Returns a KnnVectorsReader to read the binary quantized vectors from the index.
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

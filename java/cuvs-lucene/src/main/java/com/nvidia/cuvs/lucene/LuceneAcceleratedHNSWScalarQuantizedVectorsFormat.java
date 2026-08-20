/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;

import java.io.IOException;
import java.util.logging.Logger;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

/**
 * cuVS based Scalar Quantized KnnVectorsFormat for indexing on GPU and searching on the CPU.
 *
 * @since 26.02
 */
public class LuceneAcceleratedHNSWScalarQuantizedVectorsFormat extends KnnVectorsFormat {

  private static final Logger log =
      Logger.getLogger(LuceneAcceleratedHNSWScalarQuantizedVectorsFormat.class.getName());
  private static final int MAX_DIMENSIONS = 4096;
  private static volatile FlatVectorsFormat cachedFlatVectorsFormat;

  private final AcceleratedHNSWParams acceleratedHNSWParams;
  private volatile KnnVectorsFormat cachedFallbackFormat;

  private static LuceneProvider getLuceneProvider() throws IOException {
    try {
      return LuceneProvider.getInstance(LuceneProvider.LUCENE_99_FORMAT_VERSION);
    } catch (ClassNotFoundException e) {
      throw new IOException("Lucene99 vector formats are not available in this runtime", e);
    }
  }

  private static FlatVectorsFormat getOrCreateFlatVectorsFormat() throws IOException {
    FlatVectorsFormat format = cachedFlatVectorsFormat;
    if (format == null) {
      synchronized (LuceneAcceleratedHNSWScalarQuantizedVectorsFormat.class) {
        format = cachedFlatVectorsFormat;
        if (format == null) {
          try {
            format = getLuceneProvider().getLuceneScalarQuantizedVectorsFormatInstance();
            cachedFlatVectorsFormat = format;
          } catch (Exception e) {
            throw new IOException("Unable to initialize the scalar quantized flat format", e);
          }
        }
      }
    }
    return format;
  }

  private KnnVectorsFormat getOrCreateFallbackFormat() throws IOException {
    KnnVectorsFormat format = cachedFallbackFormat;
    if (format == null) {
      synchronized (this) {
        format = cachedFallbackFormat;
        if (format == null) {
          try {
            format =
                getLuceneProvider()
                    .getLuceneHnswScalarQuantizedKnnVectorsFormatInstance(
                        acceleratedHNSWParams.getMaxConn(), acceleratedHNSWParams.getBeamWidth());
            cachedFallbackFormat = format;
          } catch (Exception e) {
            throw new IOException("Unable to initialize the scalar quantized fallback format", e);
          }
        }
      }
    }
    return format;
  }

  /** Initializes {@link LuceneAcceleratedHNSWScalarQuantizedVectorsFormat} with default values. */
  public LuceneAcceleratedHNSWScalarQuantizedVectorsFormat() {
    this(new AcceleratedHNSWParams.Builder().build());
  }

  /**
   * Initializes {@link LuceneAcceleratedHNSWScalarQuantizedVectorsFormat} with the given threads, graph degree, etc.
   *
   * @param acceleratedHNSWParams An instance of {@link AcceleratedHNSWParams}
   */
  public LuceneAcceleratedHNSWScalarQuantizedVectorsFormat(
      AcceleratedHNSWParams acceleratedHNSWParams) {
    super("Lucene99AcceleratedHNSWScalarQuantizedVectorsFormat");
    this.acceleratedHNSWParams = acceleratedHNSWParams;
  }

  /**
   * Returns a KnnVectorsWriter to write the scalar quantized vectors to the index.
   */
  @Override
  public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    if (isSupported()) {
      var flatWriter = getOrCreateFlatVectorsFormat().fieldsWriter(state);
      log.info("cuVS is supported so using the Lucene99AcceleratedHNSWQuantizedVectorsWriter");
      return new LuceneAcceleratedHNSWScalarQuantizedVectorsWriter(
          state, acceleratedHNSWParams, flatWriter);
    } else {
      // Fallback to Lucene's Lucene99HnswScalarQuantizedVectorsFormat
      log.warning(
          "GPU based indexing not supported, falling back to using the"
              + " Lucene99HnswScalarQuantizedVectorsFormat");
      return getOrCreateFallbackFormat().fieldsWriter(state);
    }
  }

  /**
   * Returns a KnnVectorsReader to read the scalar quantized vectors from the index.
   */
  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    try {
      return getLuceneProvider()
          .getLuceneHnswVectorsReaderInstance(
              state, getOrCreateFlatVectorsFormat().fieldsReader(state));
    } catch (Exception e) {
      throw new IOException("Unable to initialize the scalar quantized vectors reader", e);
    }
  }

  /**
   * Returns the maximum number of vector dimensions supported by this Codec for the given field name.
   */
  @Override
  public int getMaxDimensions(String fieldName) {
    return MAX_DIMENSIONS;
  }
}

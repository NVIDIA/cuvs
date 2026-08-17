/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  private static final int MAX_DIMENSIONS = 4096;
  private static final LuceneProvider LUCENE_99_PROVIDER =
      getLuceneProvider(LuceneProvider.LUCENE_99_FORMAT_VERSION);
  private static volatile FlatVectorsFormat cachedFlatVectorsFormat;

  private final AcceleratedHNSWParams acceleratedHNSWParams;
  private volatile KnnVectorsFormat cachedFallbackFormat;

  private static LuceneProvider getLuceneProvider(String version) {
    try {
      return LuceneProvider.getInstance(version);
    } catch (Exception e) {
      throw new UnsupportedOperationException(
          "Lucene" + version + " vector formats are not available in this runtime", e);
    }
  }

  private static FlatVectorsFormat getOrCreateFlatVectorsFormat() {
    FlatVectorsFormat format = cachedFlatVectorsFormat;
    if (format == null) {
      synchronized (LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.class) {
        format = cachedFlatVectorsFormat;
        if (format == null) {
          try {
            format =
                getLuceneProvider(LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION)
                    .getLuceneBinaryQuantizedVectorsFormatInstance();
            cachedFlatVectorsFormat = format;
          } catch (Exception e) {
            throw new UnsupportedOperationException(
                "Binary quantized vectors require Lucene102 vector formats", e);
          }
        }
      }
    }
    return format;
  }

  private KnnVectorsFormat getOrCreateFallbackFormat() throws Exception {
    KnnVectorsFormat format = cachedFallbackFormat;
    if (format == null) {
      synchronized (this) {
        format = cachedFallbackFormat;
        if (format == null) {
          format =
              getLuceneProvider(LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION)
                  .getLuceneHnswBinaryQuantizedKnnVectorsFormatInstance(
                      acceleratedHNSWParams.getMaxConn(), acceleratedHNSWParams.getBeamWidth());
          cachedFallbackFormat = format;
        }
      }
    }
    return format;
  }

  /**
   * Initializes {@link LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat} with default values.
   *
   * @throws LibraryException if the native library fails to load
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
    if (isSupported()) {
      var flatWriter = getOrCreateFlatVectorsFormat().fieldsWriter(state);
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
        return getOrCreateFallbackFormat().fieldsWriter(state);
      } catch (Exception e) {
        throw new IOException("Unable to initialize the binary quantized fallback writer", e);
      }
    }
  }

  /**
   * Returns a KnnVectorsReader to read the binary quantized vectors from the index.
   */
  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    try {
      return LUCENE_99_PROVIDER.getLuceneHnswVectorsReaderInstance(
          state, getOrCreateFlatVectorsFormat().fieldsReader(state));
    } catch (Exception e) {
      throw new IOException("Unable to initialize the binary quantized vectors reader", e);
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

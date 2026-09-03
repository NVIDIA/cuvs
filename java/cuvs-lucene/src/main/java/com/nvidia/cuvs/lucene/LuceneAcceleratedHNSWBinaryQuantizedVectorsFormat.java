/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;

import com.nvidia.cuvs.LibraryException;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.concurrent.Callable;
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
  private static volatile FlatVectorsFormat cachedFlatVectorsFormat;

  private final AcceleratedHNSWParams acceleratedHNSWParams;
  private volatile KnnVectorsFormat cachedFallbackFormat;

  private static LuceneProvider getLucene99Provider() throws IOException {
    try {
      return LuceneProvider.getInstance(LuceneProvider.LUCENE_99_FORMAT_VERSION);
    } catch (ClassNotFoundException e) {
      throw new IOException("Lucene99 vector formats are not available in this runtime", e);
    }
  }

  private static RuntimeException handleConstructionFailure(String formatName, Throwable failure)
      throws IOException {
    if (failure instanceof IOException
        || failure instanceof RuntimeException
        || failure instanceof Error) {
      return Utils.handleThrowable(failure);
    }
    return new IllegalStateException("Unable to construct " + formatName, failure);
  }

  static <T> T constructLucene102Format(String formatName, Callable<T> constructor)
      throws IOException {
    try {
      return constructor.call();
    } catch (ClassNotFoundException e) {
      throw new UnsupportedOperationException(
          formatName + " is not available in this Lucene runtime", e);
    } catch (InvocationTargetException e) {
      throw handleConstructionFailure(formatName, e.getTargetException());
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException("Unable to construct " + formatName, e);
    } catch (IOException | RuntimeException | Error e) {
      throw Utils.handleThrowable(e);
    } catch (Exception e) {
      throw new IllegalStateException("Unable to construct " + formatName, e);
    }
  }

  private static FlatVectorsFormat getOrCreateFlatVectorsFormat() throws IOException {
    FlatVectorsFormat format = cachedFlatVectorsFormat;
    if (format == null) {
      synchronized (LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.class) {
        format = cachedFlatVectorsFormat;
        if (format == null) {
          format =
              constructLucene102Format(
                  "Lucene102BinaryQuantizedVectorsFormat",
                  () ->
                      LuceneProvider.getInstance(LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION)
                          .getLuceneBinaryQuantizedVectorsFormatInstance());
          cachedFlatVectorsFormat = format;
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
          format =
              constructLucene102Format(
                  "Lucene102HnswBinaryQuantizedVectorsFormat",
                  () ->
                      LuceneProvider.getInstance(LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION)
                          .getLuceneHnswBinaryQuantizedKnnVectorsFormatInstance(
                              acceleratedHNSWParams.getMaxConn(),
                              acceleratedHNSWParams.getBeamWidth()));
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
      // Fallback to Lucene's Lucene102HnswBinaryQuantizedVectorsFormat format
      log.log(
          Level.WARNING,
          "GPU based indexing not supported, falling back to using the"
              + " Lucene102HnswBinaryQuantizedVectorsFormat");
      return getOrCreateFallbackFormat().fieldsWriter(state);
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

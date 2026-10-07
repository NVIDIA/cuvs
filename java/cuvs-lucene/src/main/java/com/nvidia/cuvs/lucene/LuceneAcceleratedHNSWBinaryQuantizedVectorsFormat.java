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
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentInfo;
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
  private static volatile FlatVectorsFormat cachedLegacyFlatVectorsFormat;
  private static volatile FlatVectorsFormat cachedBinaryFlatVectorsFormat;
  private static final int GPU_MAX_DIMENSIONS = 4096;
  private static final String LUCENE_102_BINARY_META_EXTENSION = "vemb";
  private static final String LUCENE_102_BINARY_DATA_EXTENSION = "veb";

  static final String FLAT_LAYOUT_ATTRIBUTE_KEY =
      "com.nvidia.cuvs.lucene.binary_quantized_flat_layout";
  static final String LUCENE_102_BINARY_FLAT_LAYOUT = "lucene102-binary-v1";

  private final AcceleratedHNSWParams acceleratedHNSWParams;

  private static LuceneProvider getLucene99Provider() throws IOException {
    try {
      return LuceneProvider.getInstance(LuceneProvider.LUCENE_99_FORMAT_VERSION);
    } catch (ClassNotFoundException e) {
      throw new IOException("Lucene99 vector formats are not available in this runtime", e);
    }
  }

  private static LuceneProvider getLucene102Provider() throws IOException {
    try {
      return LuceneProvider.getInstance(LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION);
    } catch (ClassNotFoundException e) {
      throw new IOException("Lucene102 vector formats are not available in this runtime", e);
    }
  }

  private static FlatVectorsFormat getOrCreateLegacyFlatVectorsFormat() throws IOException {
    FlatVectorsFormat format = cachedLegacyFlatVectorsFormat;
    if (format == null) {
      synchronized (LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.class) {
        format = cachedLegacyFlatVectorsFormat;
        if (format == null) {
          try {
            format =
                getLucene99Provider()
                    .getLuceneFlatVectorsFormatInstance(DefaultFlatVectorScorer.INSTANCE);
            cachedLegacyFlatVectorsFormat = format;
          } catch (Exception e) {
            throw Utils.handleThrowable(e);
          }
        }
      }
    }
    return format;
  }

  private static FlatVectorsFormat getOrCreateBinaryFlatVectorsFormat() throws IOException {
    FlatVectorsFormat format = cachedBinaryFlatVectorsFormat;
    if (format == null) {
      synchronized (LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.class) {
        format = cachedBinaryFlatVectorsFormat;
        if (format == null) {
          try {
            format = getLucene102Provider().getLuceneBinaryQuantizedVectorsFormatInstance();
            cachedBinaryFlatVectorsFormat = format;
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
    if (isSupported()) {
      markLucene102BinaryLayout(state.segmentInfo);
      var flatWriter = getOrCreateBinaryFlatVectorsFormat().fieldsWriter(state);
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
        markLucene102BinaryLayout(state.segmentInfo);
        KnnVectorsFormat fallbackFormat =
            getLucene102Provider()
                .getLuceneHnswBinaryQuantizedKnnVectorsFormatInstance(
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
      if (usesLucene102BinaryLayout(state)) {
        KnnVectorsFormat fallbackFormat =
            getLucene102Provider()
                .getLuceneHnswBinaryQuantizedKnnVectorsFormatInstance(
                    acceleratedHNSWParams.getMaxConn(), acceleratedHNSWParams.getBeamWidth());
        return fallbackFormat.fieldsReader(state);
      }
      return getLucene99Provider()
          .getLuceneHnswVectorsReaderInstance(
              state, getOrCreateLegacyFlatVectorsFormat().fieldsReader(state));
    } catch (Exception e) {
      throw Utils.handleThrowable(e);
    }
  }

  static void markLucene102BinaryLayout(SegmentInfo segmentInfo) {
    String current = segmentInfo.getAttribute(FLAT_LAYOUT_ATTRIBUTE_KEY);
    if (current == null) {
      segmentInfo.putAttribute(FLAT_LAYOUT_ATTRIBUTE_KEY, LUCENE_102_BINARY_FLAT_LAYOUT);
    } else if (LUCENE_102_BINARY_FLAT_LAYOUT.equals(current) == false) {
      throw new IllegalStateException(
          "Segment "
              + segmentInfo.name
              + " already declares an incompatible binary flat-vector layout: "
              + current);
    }
  }

  private static boolean usesLucene102BinaryLayout(SegmentReadState state) throws IOException {
    String layout = state.segmentInfo.getAttribute(FLAT_LAYOUT_ATTRIBUTE_KEY);
    String[] segmentFiles = state.directory.listAll();
    boolean hasBinaryMetadata =
        hasSegmentFile(state, segmentFiles, LUCENE_102_BINARY_META_EXTENSION);
    boolean hasBinaryData = hasSegmentFile(state, segmentFiles, LUCENE_102_BINARY_DATA_EXTENSION);

    if (hasBinaryMetadata != hasBinaryData) {
      throw corruptLayout(
          state,
          "Lucene102 binary flat-vector files are incomplete: metadata="
              + hasBinaryMetadata
              + ", data="
              + hasBinaryData);
    }
    if (layout == null) {
      if (hasBinaryMetadata) {
        throw corruptLayout(
            state,
            "Lucene102 binary flat-vector files are present without the required layout marker");
      }
      return false;
    }
    if (LUCENE_102_BINARY_FLAT_LAYOUT.equals(layout) == false) {
      throw corruptLayout(state, "Unsupported binary flat-vector layout marker: " + layout);
    }
    if (hasBinaryMetadata == false) {
      throw corruptLayout(
          state,
          "The layout marker declares Lucene102 binary flat vectors, but their files are absent");
    }
    return true;
  }

  private static boolean hasSegmentFile(
      SegmentReadState state, String[] segmentFiles, String extension) {
    String expected =
        IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, extension);
    for (String file : segmentFiles) {
      if (expected.equals(file)) {
        return true;
      }
    }
    return false;
  }

  private static CorruptIndexException corruptLayout(SegmentReadState state, String message) {
    return new CorruptIndexException(
        message
            + "; attribute="
            + FLAT_LAYOUT_ATTRIBUTE_KEY
            + ", segmentSuffix="
            + state.segmentSuffix,
        state.segmentInfo.name);
  }

  /**
   * Returns the maximum number of vector dimensions supported by this codec for the given field name.
   *
   * <p>Returns 4096 when cuVS is supported for the current thread. Otherwise, returns {@link
   * KnnVectorsFormat#DEFAULT_MAX_DIMENSIONS}, which is 1024 in the targeted Lucene version, for the
   * CPU fallback.
   */
  @Override
  public int getMaxDimensions(String fieldName) {
    // The accelerated writer supports wider vectors than Lucene's CPU fallback formats.
    return isSupported() ? GPU_MAX_DIMENSIONS : KnnVectorsFormat.DEFAULT_MAX_DIMENSIONS;
  }
}

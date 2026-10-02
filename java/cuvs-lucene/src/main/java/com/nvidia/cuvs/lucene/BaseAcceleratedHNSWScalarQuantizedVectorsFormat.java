/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;

import java.io.IOException;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.logging.Logger;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

/**
 * Base class of the cuVS based scalar-quantized formats, which build the graph on the GPU and write
 * it in Lucene's HNSW format for search on the CPU. The formats differ in the Lucene flat format
 * that stores the quantized vectors.
 */
abstract class BaseAcceleratedHNSWScalarQuantizedVectorsFormat extends KnnVectorsFormat {

  private static final Logger log =
      Logger.getLogger(BaseAcceleratedHNSWScalarQuantizedVectorsFormat.class.getName());
  private static final int MAX_DIMENSIONS = 4096;

  private final AcceleratedHNSWParams acceleratedHNSWParams;
  private final Supplier<FlatVectorsFormat> flatVectorsFormat;
  private final Function<AcceleratedHNSWParams, KnnVectorsFormat> cpuFormat;

  /**
   * Initializes the format.
   *
   * @param name the format's name
   * @param acceleratedHNSWParams the index build parameters
   * @param flatVectorsFormat creates the format that stores the quantized vectors
   * @param cpuFormat returns the Lucene format to build the graph with when no GPU is available;
   *     it has to store the vectors with {@code flatVectorsFormat}
   *     <p>Both are only called when the format is used, never while Lucene's service loader creates
   *     it, so pass lambdas rather than method references to {@code LuceneCompat}, which may not
   *     link against the Lucene release in use (see {@link CuVSFilterCodec#delegate}).
   */
  BaseAcceleratedHNSWScalarQuantizedVectorsFormat(
      String name,
      AcceleratedHNSWParams acceleratedHNSWParams,
      Supplier<FlatVectorsFormat> flatVectorsFormat,
      Function<AcceleratedHNSWParams, KnnVectorsFormat> cpuFormat) {
    super(name);
    this.acceleratedHNSWParams = acceleratedHNSWParams;
    this.flatVectorsFormat = flatVectorsFormat;
    this.cpuFormat = cpuFormat;
  }

  /**
   * Returns a KnnVectorsWriter to write the scalar quantized vectors to the index.
   */
  @Override
  public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    LuceneVersionGuard.ensureCompatible();
    String readOnlyReason = readOnlyReason();
    if (readOnlyReason != null) {
      throw new UnsupportedOperationException(readOnlyReason);
    }
    if (isSupported()) {
      log.fine("cuVS is supported so using the Lucene99AcceleratedHNSWQuantizedVectorsWriter");
      return new LuceneAcceleratedHNSWScalarQuantizedVectorsWriter(
          state, acceleratedHNSWParams, flatVectorsFormat.get().fieldsWriter(state));
    } else {
      KnnVectorsFormat fallback = cpuFormat.apply(acceleratedHNSWParams);
      // The class name, not getName(): in Lucene 10.4 and 10.5,
      // Lucene104HnswScalarQuantizedVectorsFormat
      // reports the name of its binary-quantized sibling.
      log.warning(
          "GPU based indexing not supported, falling back to using the "
              + fallback.getClass().getSimpleName());
      return fallback.fieldsWriter(state);
    }
  }

  /**
   * Returns why this format cannot write on the running Lucene release, or {@code null} if it can.
   * Checked before writing, so that the error names the format and its replacement rather than
   * coming from Lucene's backward codecs.
   */
  String readOnlyReason() {
    return null;
  }

  /**
   * Returns a KnnVectorsReader to read the scalar quantized vectors from the index.
   */
  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    LuceneVersionGuard.ensureCompatible();
    return new Lucene99HnswVectorsReader(state, flatVectorsFormat.get().fieldsReader(state));
  }

  /**
   * Returns the maximum number of vector dimensions supported by this Codec for the given field name.
   */
  @Override
  public int getMaxDimensions(String fieldName) {
    return MAX_DIMENSIONS;
  }
}

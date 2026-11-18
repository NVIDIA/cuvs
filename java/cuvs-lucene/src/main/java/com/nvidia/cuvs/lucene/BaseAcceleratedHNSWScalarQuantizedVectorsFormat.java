/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;

import java.io.IOException;
import java.util.function.Supplier;
import java.util.logging.Logger;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.util.Version;

/**
 * Base class of the cuVS based scalar-quantized formats, which build the graph on the GPU and write
 * it in Lucene's HNSW format for search on the CPU. The formats differ only in the Lucene flat
 * format that stores the quantized vectors.
 */
abstract class BaseAcceleratedHNSWScalarQuantizedVectorsFormat extends KnnVectorsFormat {

  private static final int MAX_DIMENSIONS = 4096;

  private final String flatFormatName;
  private final Logger log;
  private final AcceleratedHNSWParams acceleratedHNSWParams;
  private final Supplier<FlatVectorsFormat> flatVectorsFormatSupplier;
  // Created on first use and then reused; see flatVectorsFormat().
  private volatile FlatVectorsFormat flatVectorsFormat;

  /**
   * Initializes the format.
   *
   * @param name the format's name
   * @param flatFormatName the name of the Lucene flat format, for error messages
   * @param log the public subclass's logger, held by it, so that logging configured for a format
   *     keeps working
   * @param acceleratedHNSWParams the index build parameters
   * @param flatVectorsFormat creates the format that stores the quantized vectors. Pass a lambda
   *     calling {@link LuceneCompat}, not a method reference, for the reason given in {@link
   *     #flatVectorsFormat()}.
   */
  BaseAcceleratedHNSWScalarQuantizedVectorsFormat(
      String name,
      String flatFormatName,
      Logger log,
      AcceleratedHNSWParams acceleratedHNSWParams,
      Supplier<FlatVectorsFormat> flatVectorsFormat) {
    super(name);
    this.flatFormatName = flatFormatName;
    this.log = log;
    this.acceleratedHNSWParams = acceleratedHNSWParams;
    this.flatVectorsFormatSupplier = flatVectorsFormat;
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
    FlatVectorsFormat flatFormat = flatVectorsFormat();
    var flatWriter = flatFormat.fieldsWriter(state);
    if (isSupported()) {
      log.info(getName() + ": cuVS is supported, so building the graph on the GPU");
      return new LuceneAcceleratedHNSWScalarQuantizedVectorsWriter(
          state, acceleratedHNSWParams, flatWriter);
    } else {
      // Builds the graph on the CPU over the scalar-quantized flat writer opened above, which is
      // what Lucene's HNSW scalar-quantized formats write, so the reader below reads it. On Lucene
      // 10.5, Lucene's writer also reads merged vectors back with flatFormat to build the graph.
      // Starts with the prefix the user guide documents for the fallback warning.
      log.warning(
          "GPU based indexing not supported, falling back to using the Lucene99HnswVectorsWriter"
              + " for "
              + getName());
      return LuceneCompat.hnswVectorsWriter(
          state,
          acceleratedHNSWParams.getMaxConn(),
          acceleratedHNSWParams.getBeamWidth(),
          flatFormat,
          flatWriter,
          acceleratedHNSWParams.getNumMergeWorkers(),
          new TaskExecutor(acceleratedHNSWParams.getMergeExec()));
    }
  }

  /**
   * Returns why this format cannot write on the running Lucene release, or {@code null} if it can.
   * A scalar-quantized format writes only while it is the one {@link
   * CuVSCodecs#acceleratedHNSWScalarQuantizedFormat} returns: once Lucene moves its flat format to
   * the backward codecs, a new format replaces it. Checked before writing, so that the error names
   * the format and its replacement rather than coming from Lucene's backward codecs.
   */
  String readOnlyReason() {
    String current = currentFormatName();
    if (current.equals(getName())) {
      return null;
    }
    // The replacement is named by its registered name, which a per-field configuration needs.
    return getName()
        + " can only read indexes on Lucene "
        + Version.LATEST
        + ", which no longer writes "
        + flatFormatName
        + ". Write with "
        + current
        + ", which CuVSCodecs.acceleratedHNSWScalarQuantizedFormat returns, or with the codec"
        + " CuVSCodecs.acceleratedHNSWScalarQuantized returns, whose class also takes a delegate"
        + " codec. A codec of your own that switches to it needs a new codec name, and the old"
        + " codec has to stay registered so that the segments it wrote stay readable.";
  }

  /**
   * Returns the name of the scalar-quantized format that writes on the running Lucene release, the
   * one {@link CuVSCodecs#acceleratedHNSWScalarQuantizedFormat} returns.
   */
  static String currentFormatName() {
    return LuceneCompat.SCALAR_QUANTIZED_FORMAT_NAME;
  }

  /**
   * Returns a KnnVectorsReader to read the scalar quantized vectors from the index.
   */
  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    LuceneVersionGuard.ensureCompatible();
    return new Lucene99HnswVectorsReader(
        state, flatVectorsFormat().fieldsReader(LuceneCompat.flatReadState(state)));
  }

  /**
   * Returns the format that stores the quantized vectors, creating it on first use.
   *
   * <p>Not created when this format is: Lucene's service loader creates every registered format
   * eagerly, and on a Lucene release that moved the flat format or does not have it, creating it
   * there would break every format and codec lookup in the JVM before {@link LuceneVersionGuard}
   * could report the mismatch. It is created through {@link LuceneCompat}, which is not loaded, and
   * so not linked, until then.
   */
  private FlatVectorsFormat flatVectorsFormat() {
    FlatVectorsFormat format = flatVectorsFormat;
    if (format == null) {
      // Flat formats hold no state, so two threads racing here at most create a spare one.
      format = flatVectorsFormatSupplier.get();
      flatVectorsFormat = format;
    }
    return format;
  }

  /**
   * Returns the maximum number of vector dimensions supported by this format for the given field
   * name.
   */
  @Override
  public int getMaxDimensions(String fieldName) {
    return MAX_DIMENSIONS;
  }
}

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
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

/**
 * cuVS based Binary Quantized KnnVectorsFormat for indexing on GPU and searching on the CPU.
 *
 * <p>The vectors are binary quantized only to build the graph on the GPU. The index stores them in
 * full precision, with Lucene's {@code Lucene99FlatVectorsFormat}, and search scores them in full
 * precision too.
 *
 * @since 26.02
 */
public class LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat extends KnnVectorsFormat {

  private static final Logger log =
      Logger.getLogger(LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.class.getName());
  private static final FlatVectorsFormat FLAT_VECTORS_FORMAT =
      new Lucene99FlatVectorsFormat(DefaultFlatVectorScorer.INSTANCE);
  private static final int MAX_DIMENSIONS = 4096;

  private final AcceleratedHNSWParams acceleratedHNSWParams;

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
    LuceneVersionGuard.ensureCompatible();
    if (isSupported()) {
      log.log(
          Level.FINE,
          "cuVS is supported so using the Lucene99AcceleratedHNSWBinaryQuantizedVectorsWriter");
      return new LuceneAcceleratedHNSWBinaryQuantizedVectorsWriter(
          state, acceleratedHNSWParams, FLAT_VECTORS_FORMAT.fieldsWriter(state));
    } else {
      // The GPU writer quantizes only to build the graph and stores full-precision vectors, which
      // is exactly what Lucene's plain HNSW format writes.
      log.log(
          Level.WARNING,
          "GPU based indexing not supported, falling back to using the Lucene99HnswVectorsWriter");
      return Utils.cpuHnswFormat(acceleratedHNSWParams).fieldsWriter(state);
    }
  }

  /**
   * Returns a KnnVectorsReader to read the binary quantized vectors from the index.
   */
  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    LuceneVersionGuard.ensureCompatible();
    return new Lucene99HnswVectorsReader(state, FLAT_VECTORS_FORMAT.fieldsReader(state));
  }

  /**
   * Returns the maximum number of vector dimensions supported by this codec for the given field name.
   */
  @Override
  public int getMaxDimensions(String fieldName) {
    return MAX_DIMENSIONS;
  }
}

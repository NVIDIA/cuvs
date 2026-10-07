/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader;
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
  private static final FlatVectorsFormat FLAT_VECTORS_FORMAT =
      new Lucene99FlatVectorsFormat(DefaultFlatVectorScorer.INSTANCE);
  private static final int MAX_DIMENSIONS = 4096;
  private final AcceleratedHNSWParams acceleratedHNSWParams;

  static final String HNSW_META_CODEC_NAME = "Lucene99HnswVectorsFormatMeta";
  // The version the GPU writers record in the HNSW files. AcceleratedHNSWUtils.writeGraph writes
  // plain vInt-encoded neighbor lists, the encoding of Lucene99HnswVectorsFormat.VERSION_START.
  // Lucene 10.3 added a group-varint encoding and made it VERSION_CURRENT, so the version has to be
  // pinned to the encoding actually written.
  static final int HNSW_VERSION = Lucene99HnswVectorsFormat.VERSION_START;
  static final String HNSW_META_CODEC_EXT = "vem";
  static final String HNSW_INDEX_CODEC_NAME = "Lucene99HnswVectorsFormatIndex";
  static final String HNSW_INDEX_EXT = "vex";

  /**
   * Initializes {@link Lucene99AcceleratedHNSWVectorsFormat} with an instance
   * of {@link AcceleratedHNSWParams} with default parameter values.
   *
   * @throws LibraryException if the native library fails to load
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
    LuceneVersionGuard.ensureCompatible();
    if (isSupported()) {
      log.log(Level.FINE, "cuVS is supported so using the Lucene99AcceleratedHNSWVectorsWriter");
      return new Lucene99AcceleratedHNSWVectorsWriter(
          state, acceleratedHNSWParams, FLAT_VECTORS_FORMAT.fieldsWriter(state));
    } else {
      log.log(
          Level.WARNING,
          "GPU based indexing not supported, falling back to using the Lucene99HnswVectorsWriter");
      return Utils.cpuHnswFormat(acceleratedHNSWParams).fieldsWriter(state);
    }
  }

  /**
   * Returns a KnnVectorsReader to read the vectors from the index.
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

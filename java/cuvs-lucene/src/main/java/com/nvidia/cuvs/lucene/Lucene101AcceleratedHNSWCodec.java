/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.LibraryException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;

/**
 * A codec that enables GPU-based accelerated HNSW capability and can be used
 * to accelerated indexing using GPUs and search using CPUs. Fallbacks to CPU
 * based indexing when used on a machine without a GPU and/or cuVS.
 *
 * @since 25.10
 */
public class Lucene101AcceleratedHNSWCodec extends FilterCodec {

  private static final Logger log = Logger.getLogger(Lucene101AcceleratedHNSWCodec.class.getName());
  private static final String NAME = "Lucene101AcceleratedHNSWCodec";
  private KnnVectorsFormat format;

  /**
   * Default constructor for {@link Lucene101AcceleratedHNSWCodec}.
   *
   * @throws Exception
   */
  public Lucene101AcceleratedHNSWCodec() throws Exception {
    this(NAME, LuceneProvider.getCodec("101"));
  }

  /**
   * Constructor for {@link Lucene101AcceleratedHNSWCodec}.
   *
   * @param name the codec's name
   * @param delegate the delegate codec to filter
   */
  public Lucene101AcceleratedHNSWCodec(String name, Codec delegate) {
    super(name, delegate);
    initializeFormatDefaultValues();
  }

  /**
   * Constructor for {@link Lucene101AcceleratedHNSWCodec}.
   *
   * @param acceleratedHNSWParams instance of {@link AcceleratedHNSWParams}
   * @throws Exception exception
   */
  public Lucene101AcceleratedHNSWCodec(AcceleratedHNSWParams acceleratedHNSWParams)
      throws Exception {
    this(NAME, LuceneProvider.getCodec("101"));
    initializeFormat(acceleratedHNSWParams, 0);
  }

  /**
   * Constructor for {@link Lucene101AcceleratedHNSWCodec} used for the native flat-buffered build
   * path (see {@link NativeFlatBufferedHNSWVectorsWriter}). Package-private: only {@link
   * CagraHnswBulkIndexWriter} can guarantee the invariants (single unmerged segment, no index
   * sort, exact vector count) that native flat buffering requires, so this overload is not part
   * of the public API.
   *
   * @param acceleratedHNSWParams instance of {@link AcceleratedHNSWParams}
   * @param numInputVectors the exact number of vectors to be indexed, used to pre-size the native
   *     flat buffer
   * @throws Exception exception
   */
  Lucene101AcceleratedHNSWCodec(AcceleratedHNSWParams acceleratedHNSWParams, int numInputVectors)
      throws Exception {
    this(NAME, LuceneProvider.getCodec("101"));
    initializeFormat(acceleratedHNSWParams, numInputVectors);
  }

  /**
   * Initialize an instance of {@link Lucene99AcceleratedHNSWVectorsFormat}
   * with an instance of {@link AcceleratedHNSWParams} with default parameter values.
   */
  private void initializeFormatDefaultValues() {
    initializeFormat(new AcceleratedHNSWParams.Builder().build(), 0);
  }

  /**
   * Initialize an instance of {@link Lucene99AcceleratedHNSWVectorsFormat}.
   *
   * @param acceleratedHNSWParams instance of {@link AcceleratedHNSWParams} to use
   * @param numInputVectors the exact number of vectors to be indexed, used to pre-size the native
   *     flat buffer (0 = disabled, the default heap-buffered path)
   */
  private void initializeFormat(AcceleratedHNSWParams acceleratedHNSWParams, int numInputVectors) {
    try {
      format = new Lucene99AcceleratedHNSWVectorsFormat(acceleratedHNSWParams, numInputVectors);
      setKnnFormat(format);
    } catch (LibraryException ex) {
      log.log(
          Level.SEVERE,
          "Couldn't load native library, possible classloader issue. " + ex.getMessage());
    }
  }

  /**
   * Get the configured {@link KnnVectorsFormat}.
   *
   * @return the instance of the {@link KnnVectorsFormat}
   */
  @Override
  public KnnVectorsFormat knnVectorsFormat() {
    return format;
  }

  /**
   * Set the {@link KnnVectorsFormat}.
   *
   * @param format the {@link KnnVectorsFormat} to set
   */
  public void setKnnFormat(KnnVectorsFormat format) {
    this.format = format;
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
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
 * CuVS based codec for GPU based vector search
 *
 * @since 25.10
 */
public class Lucene101AcceleratedHNSWCodec extends FilterCodec {

  private static final Logger log = Logger.getLogger(Lucene101AcceleratedHNSWCodec.class.getName());
  private static final String NAME = "Lucene101AcceleratedHNSWCodec";
  private KnnVectorsFormat format;

  public Lucene101AcceleratedHNSWCodec() throws Exception {
    this(NAME, LuceneProvider.getCodec("101"));
  }

  public Lucene101AcceleratedHNSWCodec(String name, Codec delegate) {
    super(name, delegate);
    initializeFormatDefaultValues();
  }

  public Lucene101AcceleratedHNSWCodec(AcceleratedHNSWParams acceleratedHNSWParams)
      throws Exception {
    this(NAME, LuceneProvider.getCodec("101"));
    initializeFormat(acceleratedHNSWParams);
  }

  private void initializeFormatDefaultValues() {
    initializeFormat(new AcceleratedHNSWParams.Builder().build());
  }

  private void initializeFormat(AcceleratedHNSWParams acceleratedHNSWParams) {
    try {
      format = new Lucene99AcceleratedHNSWVectorsFormat(acceleratedHNSWParams);
      setKnnFormat(format);
    } catch (LibraryException ex) {
      log.log(
          Level.SEVERE,
          "Couldn't load native library, possible classloader issue. " + ex.getMessage());
    }
  }

  /**
   * Get the configured {@link KnnVectorsFormat}
   *
   * @return the instance of the {@link KnnVectorsFormat}
   */
  @Override
  public KnnVectorsFormat knnVectorsFormat() {
    return format;
  }

  /**
   * Set the {@link KnnVectorsFormat}
   *
   * @param format the {@link KnnVectorsFormat} to set
   */
  public void setKnnFormat(KnnVectorsFormat format) {
    this.format = format;
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
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
 * @since 26.02
 */
public class LuceneAcceleratedHNSWScalarQuantizedCodec extends FilterCodec {

  private static final Logger log =
      Logger.getLogger(LuceneAcceleratedHNSWScalarQuantizedCodec.class.getName());
  private static final String NAME = "Lucene101AcceleratedHNSWScalarQuantizedCodec";

  private KnnVectorsFormat format;

  public LuceneAcceleratedHNSWScalarQuantizedCodec() throws Exception {
    this(NAME, LuceneProvider.getCodec("101"));
  }

  public LuceneAcceleratedHNSWScalarQuantizedCodec(String name, Codec delegate) {
    super(name, delegate);
    initializeFormatDefaultValues();
  }

  public LuceneAcceleratedHNSWScalarQuantizedCodec(AcceleratedHNSWParams acceleratedHNSWParams)
      throws Exception {
    this(NAME, LuceneProvider.getCodec("101"));
    initializeFormat(acceleratedHNSWParams);
  }

  private void initializeFormatDefaultValues() {
    initializeFormat(new AcceleratedHNSWParams.Builder().build());
  }

  private void initializeFormat(AcceleratedHNSWParams acceleratedHNSWParams) {
    try {
      format = new LuceneAcceleratedHNSWScalarQuantizedVectorsFormat(acceleratedHNSWParams);
      setKnnFormat(format);
    } catch (LibraryException ex) {
      log.log(
          Level.SEVERE,
          "Couldn't load native library, possible classloader issue. " + ex.getMessage());
    }
  }

  @Override
  public KnnVectorsFormat knnVectorsFormat() {
    return format;
  }

  public void setKnnFormat(KnnVectorsFormat format) {
    this.format = format;
  }
}

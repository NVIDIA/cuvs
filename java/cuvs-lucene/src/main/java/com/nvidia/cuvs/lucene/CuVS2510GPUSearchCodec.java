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
 * cuVS based codec for GPU based vector search
 * cuVS serialization formats are in experimental phase and hence backward compatibility cannot be guaranteed.
 *
 * @since 25.10
 */
public class CuVS2510GPUSearchCodec extends FilterCodec {

  private static final Logger log = Logger.getLogger(CuVS2510GPUSearchCodec.class.getName());
  private static final String NAME = "CuVS2510GPUSearchCodec";
  private KnnVectorsFormat format;

  /**
   * Default constructor for {@link CuVS2510GPUSearchCodec}
   *
   * @throws Exception
   */
  public CuVS2510GPUSearchCodec() throws Exception {
    this(NAME, LuceneProvider.getCodec("101"));
    initializeFormat(new GPUSearchParams.Builder().build());
  }

  /**
   * Initialize {@link CuVS2510GPUSearchCodec} with default parameter values.
   *
   * @param name the name of the codec
   * @param delegate the delegate codec
   */
  public CuVS2510GPUSearchCodec(String name, Codec delegate) {
    super(name, delegate);
    initializeFormat(new GPUSearchParams.Builder().build());
  }

  /**
   * Initialize the codec with custom parameter values.
   *
   * @param params An instance of {@link GPUSearchParams}
   * @throws Exception Exception raised when initializing the codec.
   */
  public CuVS2510GPUSearchCodec(GPUSearchParams params) throws Exception {
    this(NAME, LuceneProvider.getCodec("101"));
    initializeFormat(params);
  }

  private void initializeFormat(GPUSearchParams params) {
    try {
      format = new CuVS2510GPUVectorsFormat(params);
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

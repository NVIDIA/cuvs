/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.LibraryException;
import com.nvidia.cuvs.lucene.CuVS2510GPUVectorsWriter.IndexType;
import java.lang.reflect.InvocationTargetException;
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

  private static final int DEFAULT_CUVS_WRITER_THREADS = 1;
  private static final int DEFAULT_INTERMEDIATE_GRAPH_DEGREE = 128;
  private static final int DEFAULT_GRAPH_DEGREE = 64;
  private static final IndexType DEFAULT_INDEX_TYPE = IndexType.CAGRA;

  private KnnVectorsFormat format;

  /**
   * Default constructor for {@link CuVS2510GPUSearchCodec}
   * @throws InvocationTargetException
   * @throws IllegalArgumentException
   * @throws IllegalAccessException
   * @throws InstantiationException
   * @throws SecurityException
   * @throws NoSuchMethodException
   * @throws ClassNotFoundException
   */
  public CuVS2510GPUSearchCodec()
      throws ClassNotFoundException,
          NoSuchMethodException,
          SecurityException,
          InstantiationException,
          IllegalAccessException,
          IllegalArgumentException,
          InvocationTargetException {
    this(NAME, LuceneProvider.getCodec("101"));
  }

  /**
   * Constructor for the {@link CuVS2510GPUSearchCodec}
   *
   * @param name the name of the codec
   * @param delegate the delegate codec
   */
  public CuVS2510GPUSearchCodec(String name, Codec delegate) {
    super(name, delegate);
    try {
      format =
          new CuVS2510GPUVectorsFormat(
              DEFAULT_CUVS_WRITER_THREADS,
              DEFAULT_INTERMEDIATE_GRAPH_DEGREE,
              DEFAULT_GRAPH_DEGREE,
              DEFAULT_INDEX_TYPE);
      setKnnFormat(format);
    } catch (LibraryException ex) {
      log.severe("Couldn't load native library, possible classloader issue. " + ex.getMessage());
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

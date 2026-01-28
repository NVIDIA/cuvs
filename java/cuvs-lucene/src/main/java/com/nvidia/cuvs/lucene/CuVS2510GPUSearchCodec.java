/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.LibraryException;
import com.nvidia.cuvs.lucene.CuVS2510GPUVectorsWriter.IndexType;
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

  private static final int DEFAULT_CUVS_WRITER_THREADS = 1;
  private static final int DEFAULT_INTERMEDIATE_GRAPH_DEGREE = 128;
  private static final int DEFAULT_GRAPH_DEGREE = 64;
  private static final CagraGraphBuildAlgo DEFAULT_CAGRA_GRAPH_BUILD_ALGO =
      CagraGraphBuildAlgo.NN_DESCENT;
  private static final IndexType DEFAULT_INDEX_TYPE = IndexType.CAGRA;

  private KnnVectorsFormat format;

  /**
   * Default constructor for {@link CuVS2510GPUSearchCodec}
   *
   * @throws Exception
   */
  public CuVS2510GPUSearchCodec() throws Exception {
    this(NAME, LuceneProvider.getCodec("101"));
  }

  /**
   * Initialize {@link CuVS2510GPUSearchCodec} with default parameter values.
   *
   * @param name the name of the codec
   * @param delegate the delegate codec
   */
  public CuVS2510GPUSearchCodec(String name, Codec delegate) {
    super(name, delegate);
    initializeFormat(
        DEFAULT_INTERMEDIATE_GRAPH_DEGREE,
        DEFAULT_GRAPH_DEGREE,
        DEFAULT_CUVS_WRITER_THREADS,
        DEFAULT_CAGRA_GRAPH_BUILD_ALGO,
        DEFAULT_INDEX_TYPE);
  }

  /**
   * Initialize the codec with custom parameter values.
   *
   * @param cuvsWriterThreads The number of cuVS writer threads to use while indexing.
   * @param intermediateGraphDegree The intermediate graph degree to use while indexing.
   * @param graphDegree The graph degree to use while indexing.
   * @param cagraGraphBuildAlgo The CAGRA graph build algo to use.
   * @param indexType Type of Index to build.
   *
   * @throws Exception Exception raised when initializing the codec.
   */
  public CuVS2510GPUSearchCodec(
      int cuvsWriterThreads,
      int intermediateGraphDegree,
      int graphDegree,
      CagraGraphBuildAlgo cagraGraphBuildAlgo,
      IndexType indexType)
      throws Exception {
    this(NAME, LuceneProvider.getCodec("101"));
    initializeFormat(
        cuvsWriterThreads, intermediateGraphDegree, graphDegree, cagraGraphBuildAlgo, indexType);
  }

  private void initializeFormat(
      int cuvsWriterThreads,
      int intermediateGraphDegree,
      int graphDegree,
      CagraGraphBuildAlgo cagraGraphBuildAlgo,
      IndexType indexType) {
    try {
      format =
          new CuVS2510GPUVectorsFormat(
              cuvsWriterThreads,
              intermediateGraphDegree,
              graphDegree,
              cagraGraphBuildAlgo,
              indexType);
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

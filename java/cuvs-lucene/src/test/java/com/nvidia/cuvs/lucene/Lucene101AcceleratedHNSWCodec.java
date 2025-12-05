/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.LibraryException;
import java.lang.reflect.InvocationTargetException;
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

  private static final int DEFAULT_CUVS_WRITER_THREADS = 1;
  private static final int DEFAULT_INTERMEDIATE_GRAPH_DEGREE = 128;
  private static final int DEFAULT_GRAPH_DEGREE = 64;
  private static final int DEFAULT_HNSW_LAYERS = 1;
  private static final String NAME = "Lucene101AcceleratedHNSWCodec";
  private static final LuceneProvider lucene99Provider;
  private static final Integer maxConn;
  private static final Integer beamWidth;

  private KnnVectorsFormat format;

  static {
    try {
      lucene99Provider = LuceneProvider.getInstance("99");
      maxConn = lucene99Provider.getStaticIntParam("DEFAULT_MAX_CONN");
      beamWidth = lucene99Provider.getStaticIntParam("DEFAULT_BEAM_WIDTH");
    } catch (Exception e) {
      throw new ExceptionInInitializerError(e.getMessage());
    }
  }

  public Lucene101AcceleratedHNSWCodec()
      throws ClassNotFoundException,
          NoSuchMethodException,
          SecurityException,
          InstantiationException,
          IllegalAccessException,
          IllegalArgumentException,
          InvocationTargetException {
    this(NAME, LuceneProvider.getCodec("101"));
  }

  public Lucene101AcceleratedHNSWCodec(String name, Codec delegate) {
    super(name, delegate);
    initializeFormatDefaultValues();
  }

  public Lucene101AcceleratedHNSWCodec(
      int cuvsWriterThreads,
      int intGraphDegree,
      int graphDegree,
      int hnswLayers,
      int maxConn,
      int beamWidth)
      throws ClassNotFoundException,
          NoSuchMethodException,
          SecurityException,
          InstantiationException,
          IllegalAccessException,
          IllegalArgumentException,
          InvocationTargetException {
    this(NAME, LuceneProvider.getCodec("101"));
    initializeFormat(
        cuvsWriterThreads, intGraphDegree, graphDegree, hnswLayers, maxConn, beamWidth);
  }

  private void initializeFormatDefaultValues() {
    initializeFormat(
        DEFAULT_CUVS_WRITER_THREADS,
        DEFAULT_INTERMEDIATE_GRAPH_DEGREE,
        DEFAULT_GRAPH_DEGREE,
        DEFAULT_HNSW_LAYERS,
        maxConn,
        beamWidth);
  }

  private void initializeFormat(
      int cuvsWriterThreads,
      int intGraphDegree,
      int graphDegree,
      int hnswLayers,
      int maxConn,
      int beamWidth) {
    try {
      format =
          new Lucene99AcceleratedHNSWVectorsFormat(
              cuvsWriterThreads, intGraphDegree, graphDegree, hnswLayers, maxConn, beamWidth);
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

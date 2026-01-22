/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.DEFAULT_GRAPH_DEGREE;
import static com.nvidia.cuvs.lucene.LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.DEFAULT_HNSW_GRAPH_LAYERS;
import static com.nvidia.cuvs.lucene.LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.DEFAULT_INTERMEDIATE_GRAPH_DEGREE;
import static com.nvidia.cuvs.lucene.LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.DEFAULT_WRITER_THREADS;

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
public class LuceneAcceleratedHNSWBinaryQuantizedCodec extends FilterCodec {

  private static final Logger log =
      Logger.getLogger(LuceneAcceleratedHNSWBinaryQuantizedCodec.class.getName());
  private static final String NAME = "Lucene101AcceleratedHNSWBinaryQuantizedCodec";
  private static final LuceneProvider LUCENE99_PROVIDER;
  private static final Integer DEFAULT_MAX_CONN;
  private static final Integer DEFAULT_BEAM_WIDTH;

  private KnnVectorsFormat format;

  static {
    try {
      LUCENE99_PROVIDER = LuceneProvider.getInstance("99");
      DEFAULT_MAX_CONN = LUCENE99_PROVIDER.getStaticIntParam("DEFAULT_MAX_CONN");
      DEFAULT_BEAM_WIDTH = LUCENE99_PROVIDER.getStaticIntParam("DEFAULT_BEAM_WIDTH");
    } catch (Exception e) {
      throw new ExceptionInInitializerError(e.getMessage());
    }
  }

  public LuceneAcceleratedHNSWBinaryQuantizedCodec() throws Exception {
    this(NAME, LuceneProvider.getCodec("101"));
  }

  public LuceneAcceleratedHNSWBinaryQuantizedCodec(String name, Codec delegate) {
    super(name, delegate);
    initializeFormatDefaultValues();
  }

  public LuceneAcceleratedHNSWBinaryQuantizedCodec(
      int cuvsWriterThreads,
      int intGraphDegree,
      int graphDegree,
      int hnswLayers,
      int maxConn,
      int beamWidth)
      throws Exception {
    this(NAME, LuceneProvider.getCodec("101"));
    initializeFormat(
        cuvsWriterThreads, intGraphDegree, graphDegree, hnswLayers, maxConn, beamWidth);
  }

  private void initializeFormatDefaultValues() {
    initializeFormat(
        DEFAULT_WRITER_THREADS,
        DEFAULT_INTERMEDIATE_GRAPH_DEGREE,
        DEFAULT_GRAPH_DEGREE,
        DEFAULT_HNSW_GRAPH_LAYERS,
        DEFAULT_MAX_CONN,
        DEFAULT_BEAM_WIDTH);
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
          new LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat(
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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.DEFAULT_BEAM_WIDTH;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.DEFAULT_MAX_CONN;

import com.nvidia.cuvs.LibraryException;
import java.util.logging.Logger;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene101.Lucene101Codec;

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

  private KnnVectorsFormat format;

  public Lucene101AcceleratedHNSWCodec() {
    this(NAME, new Lucene101Codec());
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
      int beamWidth) {
    this(NAME, new Lucene101Codec());
    initializeFormat(
        cuvsWriterThreads, intGraphDegree, graphDegree, hnswLayers, maxConn, beamWidth);
  }

  private void initializeFormatDefaultValues() {
    initializeFormat(
        DEFAULT_CUVS_WRITER_THREADS,
        DEFAULT_INTERMEDIATE_GRAPH_DEGREE,
        DEFAULT_GRAPH_DEGREE,
        DEFAULT_HNSW_LAYERS,
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
          new Lucene99AcceleratedHNSWVectorsFormat(
              cuvsWriterThreads, intGraphDegree, graphDegree, hnswLayers, maxConn, beamWidth);
      setKnnFormat(format);
    } catch (LibraryException ex) {
      log.severe("Couldn't load native library, possible classloader issue. " + ex.getMessage());
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

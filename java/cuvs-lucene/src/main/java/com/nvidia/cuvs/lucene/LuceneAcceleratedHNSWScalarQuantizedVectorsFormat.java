/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;

import com.nvidia.cuvs.LibraryException;
import java.io.IOException;
import java.util.logging.Logger;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

/**
 * cuVS based Scalar Quantized KnnVectorsFormat for indexing on GPU and searching on the CPU.
 *
 * @since 26.02
 */
public class LuceneAcceleratedHNSWScalarQuantizedVectorsFormat extends KnnVectorsFormat {

  private static final Logger log =
      Logger.getLogger(LuceneAcceleratedHNSWScalarQuantizedVectorsFormat.class.getName());
  private static final LuceneProvider LUCENE_PROVIDER;
  private static final FlatVectorsFormat FLAT_VECTORS_FORMAT;
  private static final Integer DEFAULT_MAX_CONN;
  private static final Integer DEFAULT_BEAM_WIDTH;

  public static final int DEFAULT_WRITER_THREADS = 32;
  public static final int DEFAULT_INTERMEDIATE_GRAPH_DEGREE = 128;
  public static final int DEFAULT_GRAPH_DEGREE = 64;
  public static final int DEFAULT_HNSW_GRAPH_LAYERS = 2;

  private final int maxDimensions = 4096;
  private final int cuvsWriterThreads;
  private final int intGraphDegree;
  private final int graphDegree;
  private final int hnswLayers;
  private final int maxConn;
  private final int beamWidth;

  static {
    try {
      LUCENE_PROVIDER = LuceneProvider.getInstance("99");
      DEFAULT_MAX_CONN = LUCENE_PROVIDER.getStaticIntParam("DEFAULT_MAX_CONN");
      DEFAULT_BEAM_WIDTH = LUCENE_PROVIDER.getStaticIntParam("DEFAULT_BEAM_WIDTH");
      FLAT_VECTORS_FORMAT = LUCENE_PROVIDER.getLuceneScalarQuantizedVectorsFormatInstance();
    } catch (Exception e) {
      throw new ExceptionInInitializerError(e.getMessage());
    }
  }

  /**
   * Initializes {@link LuceneAcceleratedHNSWScalarQuantizedVectorsFormat} with default values.
   *
   * @throws LibraryException if the native library fails to load
   */
  public LuceneAcceleratedHNSWScalarQuantizedVectorsFormat() {
    this(
        DEFAULT_WRITER_THREADS,
        DEFAULT_INTERMEDIATE_GRAPH_DEGREE,
        DEFAULT_GRAPH_DEGREE,
        DEFAULT_HNSW_GRAPH_LAYERS,
        DEFAULT_MAX_CONN,
        DEFAULT_BEAM_WIDTH);
  }

  /**
   * Initializes {@link LuceneAcceleratedHNSWScalarQuantizedVectorsFormat} with the given threads, graph degree, etc.
   *
   * @param cuvsWriterThreads number of cuVS threads to use while building the CAGRA index
   * @param intGraphDegree the intermediate graph degree while building the CAGRA index
   * @param graphDegree the graph degree to use while building the CAGRA index
   * @param hnswLayers the number of HNSW layers to construct in the HNSW graph
   * @param maxConn the maximum connections for the HNSW graph
   * @param beamWidth the beam width to use while building the HNSW graph
   */
  public LuceneAcceleratedHNSWScalarQuantizedVectorsFormat(
      int cuvsWriterThreads,
      int intGraphDegree,
      int graphDegree,
      int hnswLayers,
      int maxConn,
      int beamWidth) {
    super("Lucene99AcceleratedHNSWScalarQuantizedVectorsFormat");

    assert cuvsWriterThreads > 0
        : "cuvsWriterThreads must be greater than zero, but is: " + cuvsWriterThreads;
    assert intGraphDegree > 0
        : "intGraphDegree must be greater than zero, but is: " + intGraphDegree;
    assert graphDegree > 0 : "graphDegree must be greater than zero, but is: " + graphDegree;
    assert hnswLayers > 0 : "hnswLayers must be greater than zero, but is: " + hnswLayers;
    assert maxConn > 0 : "maxConn must be greater than zero, but is: " + maxConn;
    assert beamWidth > 0 : "beamWidth must be greater than zero, but is: " + beamWidth;

    this.cuvsWriterThreads = cuvsWriterThreads;
    this.intGraphDegree = intGraphDegree;
    this.graphDegree = graphDegree;
    this.hnswLayers = hnswLayers;
    this.maxConn = maxConn;
    this.beamWidth = beamWidth;
  }

  /**
   * Returns a KnnVectorsWriter to write the scalar quantized vectors to the index.
   */
  @Override
  public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    var flatWriter = FLAT_VECTORS_FORMAT.fieldsWriter(state);
    if (isSupported()) {
      log.info("cuVS is supported so using the Lucene99AcceleratedHNSWQuantizedVectorsWriter");
      return new LuceneAcceleratedHNSWScalarQuantizedVectorsWriter(
          state, cuvsWriterThreads, intGraphDegree, graphDegree, hnswLayers, flatWriter);
    } else {
      try {
        // Fallback to Lucene's Lucene99HnswScalarQuantizedVectorsFormat
        log.warning(
            "GPU based indexing not supported, falling back to using the"
                + " Lucene99HnswScalarQuantizedVectorsFormat");
        KnnVectorsFormat fallbackFormat =
            LUCENE_PROVIDER.getLuceneHnswScalarQuantizedVectorsFormatInstance(beamWidth, maxConn);
        return fallbackFormat.fieldsWriter(state);
      } catch (Exception e) {
        throw new RuntimeException(e.getMessage());
      }
    }
  }

  /**
   * Returns a KnnVectorsReader to read the scalar quantized vectors from the index.
   */
  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    try {
      return LUCENE_PROVIDER.getLuceneHnswVectorsReaderInstance(
          state, FLAT_VECTORS_FORMAT.fieldsReader(state));
    } catch (Exception e) {
      throw new RuntimeException(e.getMessage());
    }
  }

  /**
   * Returns the maximum number of vector dimensions supported by this Codec for the given field name.
   */
  @Override
  public int getMaxDimensions(String fieldName) {
    return maxDimensions;
  }

  /**
   * Returns a string containing the meta information like hnsw layers, graph degree etc.
   */
  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder(this.getClass().getSimpleName());
    sb.append("(cuvsWriterThreads=").append(cuvsWriterThreads);
    sb.append("intGraphDegree=").append(intGraphDegree);
    sb.append("graphDegree=").append(graphDegree);
    sb.append("hnswLayers=").append(hnswLayers);
    sb.append(")");
    return sb.toString();
  }
}

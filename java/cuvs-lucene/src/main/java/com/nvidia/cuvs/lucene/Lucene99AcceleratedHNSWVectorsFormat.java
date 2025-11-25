/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.Utils.cuVSResourcesOrNull;

import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.LibraryException;
import java.io.IOException;
import java.util.logging.Logger;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

/**
 * cuVS based KnnVectorsFormat for indexing on GPU and searching on the CPU.
 *
 * @since 25.10
 */
public class Lucene99AcceleratedHNSWVectorsFormat extends KnnVectorsFormat {

  private static final Logger log =
      Logger.getLogger(Lucene99AcceleratedHNSWVectorsFormat.class.getName());

  static final int DEFAULT_WRITER_THREADS = 32;
  static final int DEFAULT_INTERMEDIATE_GRAPH_DEGREE = 128;
  static final int DEFAULT_GRAPH_DEGREE = 64;
  static final int DEFAULT_HNSW_GRAPH_LAYERS = 1;

  static final String HNSW_META_CODEC_NAME = "Lucene99HnswVectorsFormatMeta";
  static final String HNSW_META_CODEC_EXT = "vem";
  static final String HNSW_INDEX_CODEC_NAME = "Lucene99HnswVectorsFormatIndex";
  static final String HNSW_INDEX_EXT = "vex";
  static final LuceneProvider LUCENE_PROVIDER;

  private static CuVSResources resources = cuVSResourcesOrNull();

  private static final FlatVectorsFormat FLAT_VECTORS_FORMAT;
  private static final Integer MAX_CONN;
  private static final Integer BEAM_WIDTH;
  private static final Integer NUM_MERGE_WORKERS;

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
      MAX_CONN = LUCENE_PROVIDER.getStaticIntParam("DEFAULT_MAX_CONN");
      BEAM_WIDTH = LUCENE_PROVIDER.getStaticIntParam("DEFAULT_BEAM_WIDTH");
      NUM_MERGE_WORKERS = LUCENE_PROVIDER.getStaticIntParam("DEFAULT_BEAM_WIDTH");
      FLAT_VECTORS_FORMAT =
          LUCENE_PROVIDER.getLuceneFlatVectorsFormatInstance(DefaultFlatVectorScorer.INSTANCE);
    } catch (Exception e) {
      throw new ExceptionInInitializerError(e.getMessage());
    }
  }

  /**
   * Initializes {@link Lucene99AcceleratedHNSWVectorsFormat} with default values.
   *
   * @throws LibraryException if the native library fails to load
   */
  public Lucene99AcceleratedHNSWVectorsFormat() {
    this(
        DEFAULT_WRITER_THREADS,
        DEFAULT_INTERMEDIATE_GRAPH_DEGREE,
        DEFAULT_GRAPH_DEGREE,
        DEFAULT_HNSW_GRAPH_LAYERS,
        MAX_CONN,
        BEAM_WIDTH);
  }

  /**
   * Initializes {@link Lucene99AcceleratedHNSWVectorsFormat} with the given threads, graph degree, etc.
   *
   * @param cuvsWriterThreads number of cuVS threads to use while building the CAGRA index
   * @param intGraphDegree the intermediate graph degree while building the CAGRA index
   * @param graphDegree the graph degree to use while building the CAGRA index
   * @param hnswLayers the number of HNSW layers to construct in the HNSW graph
   * @param maxConn the maximum connections for the HNSW graph
   * @param beamWidth the beam width to use while building the HNSW graph
   */
  public Lucene99AcceleratedHNSWVectorsFormat(
      int cuvsWriterThreads,
      int intGraphDegree,
      int graphDegree,
      int hnswLayers,
      int maxConn,
      int beamWidth) {
    super("Lucene99AcceleratedHNSWVectorsFormat");
    this.cuvsWriterThreads = cuvsWriterThreads;
    this.intGraphDegree = intGraphDegree;
    this.graphDegree = graphDegree;
    this.hnswLayers = hnswLayers;
    this.maxConn = maxConn;
    this.beamWidth = beamWidth;
  }

  /**
   * Returns a KnnVectorsWriter to write the vectors to the index.
   */
  @Override
  public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    var flatWriter = FLAT_VECTORS_FORMAT.fieldsWriter(state);
    if (supported()) {
      log.info("cuVS is supported so using the Lucene99AcceleratedHNSWVectorsWriter");
      return new Lucene99AcceleratedHNSWVectorsWriter(
          state, cuvsWriterThreads, intGraphDegree, graphDegree, hnswLayers, resources, flatWriter);
    } else {
      log.warning(
          "GPU based indexing not supported, falling back to using the Lucene99HnswVectorsWriter");
      // TODO: Make num merge workers configurable.
      try {
        return LUCENE_PROVIDER.getLuceneHnswVectorsWriterInstance(
            state, maxConn, beamWidth, flatWriter, NUM_MERGE_WORKERS, null);
      } catch (Exception e) {
        // maybe there is a better suited option to throwing RuntimeException? Need to explore.
        throw new RuntimeException(e.getMessage());
      }
    }
  }

  /**
   * Returns a KnnVectorsReader to read the vectors from the index.
   */
  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    try {
      return LUCENE_PROVIDER.getLuceneHnswVectorsReaderInstance(
          state, FLAT_VECTORS_FORMAT.fieldsReader(state));
    } catch (Exception e) {
      // maybe there is a better suited option to throwing RuntimeException? Need to explore.
      throw new RuntimeException(e.getMessage());
    }
  }

  /**
   * Returns the maximum number of vector dimensions supported by this codec for the given field name.
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
    sb.append("resources=").append(resources);
    sb.append(")");
    return sb.toString();
  }

  /**
   * Gets the instance of CuVSResources
   *
   * @return the instance of CuVSResources
   */
  public static CuVSResources getResources() {
    return resources;
  }

  /**
   * Sets the instance of CuVSResources
   *
   * @param resources the instance of CuVSResources to set
   */
  public static void setResources(CuVSResources resources) {
    Lucene99AcceleratedHNSWVectorsFormat.resources = resources;
  }

  /**
   * Tells whether the platform supports cuVS.
   *
   * @return if cuVS supported or not
   */
  public static boolean supported() {
    return resources != null;
  }

  /**
   * Checks if cuVS supported and throws {@link UnsupportedOperationException} otherwise.
   */
  public static void checkSupported() {
    if (!supported()) {
      throw new UnsupportedOperationException();
    }
  }
}

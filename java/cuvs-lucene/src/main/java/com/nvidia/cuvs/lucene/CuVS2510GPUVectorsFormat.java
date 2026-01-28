/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.assertIsSupported;

import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.LibraryException;
import com.nvidia.cuvs.lucene.CuVS2510GPUVectorsWriter.IndexType;
import java.io.IOException;
import java.util.logging.Logger;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

/**
 * Extends upon the KnnVectorsFormat - Encodes/decodes per-document vector and any associated indexing structures required to support
 * GPU-based accelerated nearest-neighbor search.
 *
 * @since 25.10
 */
public class CuVS2510GPUVectorsFormat extends KnnVectorsFormat {

  static final Logger log = Logger.getLogger(CuVS2510GPUVectorsFormat.class.getName());

  static final String CUVS_META_CODEC_NAME = "Lucene102CuVSVectorsFormatMeta";
  static final String CUVS_META_CODEC_EXT = "vemc";
  static final String CUVS_INDEX_CODEC_NAME = "Lucene102CuVSVectorsFormatIndex";
  static final String CUVS_INDEX_EXT = "vcag";

  static final int VERSION_START = 0;
  static final int VERSION_CURRENT = VERSION_START;

  static final int DEFAULT_WRITER_THREADS = 32;
  static final int DEFAULT_INTERMEDIATE_GRAPH_DEGREE = 128;
  static final int DEFAULT_GRAPH_DEGREE = 64;
  static final CagraGraphBuildAlgo DEFAULT_CAGRA_GRAPH_BUILD_ALGO = CagraGraphBuildAlgo.NN_DESCENT;
  static final IndexType DEFAULT_INDEX_TYPE = IndexType.CAGRA;

  static final LuceneProvider LUCENE_PROVIDER;
  static final FlatVectorsFormat FLAT_VECTORS_FORMAT;

  final int maxDimensions = 4096;
  final int cuvsWriterThreads;
  final int intGraphDegree;
  final int graphDegree;
  final CagraGraphBuildAlgo cagraGraphBuildAlgo;
  final CuVS2510GPUVectorsWriter.IndexType indexType; // the index type to build, when writing

  static {
    try {
      LUCENE_PROVIDER = LuceneProvider.getInstance("99");
      FLAT_VECTORS_FORMAT =
          LUCENE_PROVIDER.getLuceneFlatVectorsFormatInstance(DefaultFlatVectorScorer.INSTANCE);
    } catch (Exception e) {
      throw new ExceptionInInitializerError(e.getMessage());
    }
  }

  /**
   * Initializes the {@link CuVS2510GPUVectorsFormat} with default parameter values.
   *
   * @throws LibraryException if the native library fails to load
   */
  public CuVS2510GPUVectorsFormat() {
    this(
        DEFAULT_WRITER_THREADS,
        DEFAULT_INTERMEDIATE_GRAPH_DEGREE,
        DEFAULT_GRAPH_DEGREE,
        DEFAULT_CAGRA_GRAPH_BUILD_ALGO,
        DEFAULT_INDEX_TYPE);
  }

  /**
   * Initializes the {@link CuVS2510GPUVectorsFormat} with the given threads, graph degree, etc.
   *
   * @param cuvsWriterThreads the number of cuVS writer threads to use
   * @param intGraphDegree the intermediate graph degree for building the CAGRA index
   * @param graphDegree the graph degree for building the CAGRA index
   * @param cagraGraphBuildAlgo the CAGRA graph build algorithm to use
   * @param indexType the {@link com.nvidia.cuvs.lucene.CuVS2510GPUVectorsWriter.IndexType}
   *
   * @throws LibraryException if the native library fails to load
   */
  public CuVS2510GPUVectorsFormat(
      int cuvsWriterThreads,
      int intGraphDegree,
      int graphDegree,
      CagraGraphBuildAlgo cagraGraphBuildAlgo,
      IndexType indexType) {
    super("CuVS2510GPUVectorsFormat");
    this.cuvsWriterThreads = cuvsWriterThreads;
    this.intGraphDegree = intGraphDegree;
    this.graphDegree = graphDegree;
    this.cagraGraphBuildAlgo = cagraGraphBuildAlgo;
    this.indexType = indexType;
  }

  /**
   * Returns a {@link CuVS2510GPUVectorsWriter} to write the vectors to the index.
   */
  @Override
  public CuVS2510GPUVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    assertIsSupported();
    var flatWriter = FLAT_VECTORS_FORMAT.fieldsWriter(state);
    return new CuVS2510GPUVectorsWriter(
        state,
        cuvsWriterThreads,
        intGraphDegree,
        graphDegree,
        cagraGraphBuildAlgo,
        indexType,
        flatWriter);
  }

  /**
   * Returns a KnnVectorsReader to read the vectors from the index.
   */
  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    assertIsSupported();
    return new CuVS2510GPUVectorsReader(state, FLAT_VECTORS_FORMAT.fieldsReader(state));
  }

  /**
   * Returns the maximum number of vector dimensions supported by this codec for the given field name.
   */
  @Override
  public int getMaxDimensions(String fieldName) {
    return maxDimensions;
  }

  /**
   * Returns a string containing information like cuvsWriterThreads, intGraphDegree, etc.
   */
  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder(this.getClass().getSimpleName());
    sb.append("(cuvsWriterThreads=").append(cuvsWriterThreads);
    sb.append("intGraphDegree=").append(intGraphDegree);
    sb.append("graphDegree=").append(graphDegree);
    sb.append("cagraGraphBuildAlgo=").append(cagraGraphBuildAlgo);
    sb.append(")");
    return sb.toString();
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.Utils.cuVSResourcesOrNull;

import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.LibraryException;
import com.nvidia.cuvs.lucene.CuVS2510GPUVectorsWriter.IndexType;
import java.io.IOException;
import java.util.logging.Logger;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
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
  static final IndexType DEFAULT_INDEX_TYPE = IndexType.CAGRA;

  static CuVSResources resources = cuVSResourcesOrNull();

  /** The format for storing, reading, and merging raw vectors on disk. */
  private static final FlatVectorsFormat flatVectorsFormat =
      new Lucene99FlatVectorsFormat(DefaultFlatVectorScorer.INSTANCE);

  final int maxDimensions = 4096;
  final int cuvsWriterThreads;
  final int intGraphDegree;
  final int graphDegree;
  final CuVS2510GPUVectorsWriter.IndexType indexType; // the index type to build, when writing

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
        DEFAULT_INDEX_TYPE);
  }

  /**
   * Initializes the {@link CuVS2510GPUVectorsFormat} with the given threads, graph degree, etc.
   *
   * @param cuvsWriterThreads the number of cuVS writer threads to use
   * @param intGraphDegree the intermediate graph degree for building the CAGRA index
   * @param graphDegree the graph degree for building the CAGRA index
   * @param indexType the {@link com.nvidia.cuvs.lucene.CuVS2510GPUVectorsWriter.IndexType}
   *
   * @throws LibraryException if the native library fails to load
   */
  public CuVS2510GPUVectorsFormat(
      int cuvsWriterThreads, int intGraphDegree, int graphDegree, IndexType indexType) {
    super("CuVS2510GPUVectorsFormat");
    this.cuvsWriterThreads = cuvsWriterThreads;
    this.intGraphDegree = intGraphDegree;
    this.graphDegree = graphDegree;
    this.indexType = indexType;
  }

  /**
   * Returns a {@link CuVS2510GPUVectorsWriter} to write the vectors to the index.
   */
  @Override
  public CuVS2510GPUVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    checkSupported();
    var flatWriter = flatVectorsFormat.fieldsWriter(state);
    return new CuVS2510GPUVectorsWriter(
        state, cuvsWriterThreads, intGraphDegree, graphDegree, indexType, resources, flatWriter);
  }

  /**
   * Returns a KnnVectorsReader to read the vectors from the index.
   */
  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    checkSupported();
    return new CuVS2510GPUVectorsReader(state, resources, flatVectorsFormat.fieldsReader(state));
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
    sb.append("resources=").append(resources);
    sb.append(")");
    return sb.toString();
  }

  /**
   * Tells whether the platform supports cuVS.
   *
   * @return if cuVS is supported or not
   */
  public static boolean supported() {
    return resources != null;
  }

  /**
   * Checks if cuVS is supported and throws {@link UnsupportedOperationException} otherwise.
   */
  public static void checkSupported() {
    if (!supported()) {
      throw new UnsupportedOperationException();
    }
  }
}

/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

/** CuVS based KnnVectorsFormat for GPU acceleration */
public class CuVS2510GPUVectorsFormat extends KnnVectorsFormat {

  static final Logger log = Logger.getLogger(CuVS2510GPUVectorsFormat.class.getName());

  // TODO: fix Lucene version in name, to the final targeted release, if any
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
   * Creates a CuVS2510GPUVectorsFormat, with default values.
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
   * Creates a CuVS2510GPUVectorsFormat, with the given threads, graph degree, etc.
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

  @Override
  public CuVS2510GPUVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    checkSupported();
    var flatWriter = flatVectorsFormat.fieldsWriter(state);
    return new CuVS2510GPUVectorsWriter(
        state, cuvsWriterThreads, intGraphDegree, graphDegree, indexType, resources, flatWriter);
  }

  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    checkSupported();
    return new CuVS2510GPUVectorsReader(state, resources, flatVectorsFormat.fieldsReader(state));
  }

  @Override
  public int getMaxDimensions(String fieldName) {
    return maxDimensions;
  }

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

  /** Tells whether the platform supports cuVS. */
  public static boolean supported() {
    return resources != null;
  }

  public static void checkSupported() {
    if (!supported()) {
      throw new UnsupportedOperationException();
    }
  }
}

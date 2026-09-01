/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.createMultiLayerHnswGraph;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.createSingleVectorHnswGraph;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.writeEmpty;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.writeGraph;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.writeMeta;
import static com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsFormat.HNSW_INDEX_CODEC_NAME;
import static com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsFormat.HNSW_INDEX_EXT;
import static com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsFormat.HNSW_META_CODEC_EXT;
import static com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsFormat.HNSW_META_CODEC_NAME;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.getCuVSResourcesInstance;

import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.QuantizationType;
import java.io.Closeable;
import java.io.IOException;
import java.util.List;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;

/**
 * Owns the {@code .vem}/{@code .vex} outputs and builds/writes the CAGRA-derived HNSW graph for a
 * field. Shared by both {@link Lucene99AcceleratedHNSWVectorsWriter} (heap-buffered flat path)
 * and {@link NativeFlatBufferedHNSWVectorsWriter} (native flat-buffered path) — the only
 * difference between those two writers is how the flat {@code .vec}/{@code .vemf} files and the
 * dataset handed to {@link #writeField(FieldInfo, CuVSMatrix)} are produced.
 */
final class AcceleratedHnswGraphOutput implements Closeable {

  private static final LuceneProvider LUCENE_PROVIDER;
  private static final Integer VERSION_CURRENT;

  static {
    try {
      LUCENE_PROVIDER = LuceneProvider.getInstance("99");
      VERSION_CURRENT = LUCENE_PROVIDER.getStaticIntParam("VERSION_CURRENT");
    } catch (Exception e) {
      throw new ExceptionInInitializerError(e.getMessage());
    }
  }

  private final AcceleratedHNSWParams acceleratedHNSWParams;
  private final IndexOutput hnswMeta;
  private final IndexOutput hnswVectorIndex;
  private boolean finished;

  AcceleratedHnswGraphOutput(SegmentWriteState state, AcceleratedHNSWParams acceleratedHNSWParams)
      throws IOException {
    this.acceleratedHNSWParams = acceleratedHNSWParams;
    String vemFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, HNSW_META_CODEC_EXT);
    String vexFileName =
        IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, HNSW_INDEX_EXT);
    IndexOutput meta = null;
    IndexOutput vectorIndex = null;
    boolean success = false;
    try {
      meta = state.directory.createOutput(vemFileName, state.context);
      vectorIndex = state.directory.createOutput(vexFileName, state.context);
      CodecUtil.writeIndexHeader(
          meta,
          HNSW_META_CODEC_NAME,
          VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      CodecUtil.writeIndexHeader(
          vectorIndex,
          HNSW_INDEX_CODEC_NAME,
          VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      success = true;
    } finally {
      this.hnswMeta = meta;
      this.hnswVectorIndex = vectorIndex;
      if (success == false) {
        IOUtils.closeWhileHandlingException(this);
      }
    }
  }

  /**
   * Flush/sorting path: builds a host matrix from the heap vectors, then delegates to {@link
   * #writeField(FieldInfo, CuVSMatrix)}.
   */
  void writeField(FieldInfo fieldInfo, List<float[]> vectors) throws IOException {
    if (vectors.size() == 0) {
      writeEmpty(fieldInfo, hnswMeta);
      return;
    }
    if (vectors.size() < 2) {
      writeSingleVectorGraph(fieldInfo, vectors);
      return;
    }
    CuVSMatrix dataset = Utils.createFloatMatrix(vectors, fieldInfo.getVectorDimension());
    writeField(fieldInfo, dataset);
  }

  /**
   * Builds the intermediate CAGRA index and builds and writes the HNSW index. Single
   * implementation used by both the flush and merge paths of both writers. The dataset is a
   * {@link CuVSMatrix} (host-backed on the merge/native paths) so the full set of vectors is
   * never double-materialised on the Java heap.
   */
  void writeField(FieldInfo fieldInfo, CuVSMatrix dataset) throws IOException {
    int size = (int) dataset.size();
    if (size == 0) {
      writeEmpty(fieldInfo, hnswMeta);
      return;
    }
    if (size < 2) {
      float[] buf = new float[fieldInfo.getVectorDimension()];
      dataset.getRow(0).toArray(buf);
      writeSingleVectorGraph(fieldInfo, List.of(buf));
      return;
    }
    try {
      CagraIndexParams params =
          CagraIndexParamsFactory.create(acceleratedHNSWParams, dataset.size(), dataset.columns());
      try (CagraIndex cagraIndex =
          CagraIndex.newBuilder(getCuVSResourcesInstance())
              .withDataset(dataset)
              .withIndexParams(params)
              .build()) {
        CuVSMatrix adjacencyListMatrix = cagraIndex.getGraph();
        int dimensions = fieldInfo.getVectorDimension();
        GPUBuiltHnswGraph hnswGraph =
            createMultiLayerHnswGraph(
                fieldInfo,
                dimensions,
                adjacencyListMatrix,
                dataset,
                acceleratedHNSWParams.getHnswLayers(),
                params,
                QuantizationType.NONE,
                acceleratedHNSWParams.getWriterThreads());
        long vectorIndexOffset = hnswVectorIndex.getFilePointer();
        int[][] graphLevelNodeOffsets =
            writeGraph(hnswGraph, hnswVectorIndex, acceleratedHNSWParams.getWriterThreads());
        long vectorIndexLength = hnswVectorIndex.getFilePointer() - vectorIndexOffset;
        writeMeta(
            hnswVectorIndex,
            hnswMeta,
            fieldInfo,
            vectorIndexOffset,
            vectorIndexLength,
            size,
            hnswGraph,
            graphLevelNodeOffsets);
      }
    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
  }

  private void writeSingleVectorGraph(FieldInfo fieldInfo, List<float[]> vectors)
      throws IOException {
    try {
      int size = 1;
      int dimensions = fieldInfo.getVectorDimension();
      GPUBuiltHnswGraph hnswGraph = createSingleVectorHnswGraph(size, dimensions);
      long vectorIndexOffset = hnswVectorIndex.getFilePointer();
      int[][] graphLevelNodeOffsets =
          writeGraph(hnswGraph, hnswVectorIndex, acceleratedHNSWParams.getWriterThreads());
      long vectorIndexLength = hnswVectorIndex.getFilePointer() - vectorIndexOffset;
      writeMeta(
          hnswVectorIndex,
          hnswMeta,
          fieldInfo,
          vectorIndexOffset,
          vectorIndexLength,
          size,
          hnswGraph,
          graphLevelNodeOffsets);
    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
  }

  void finish() throws IOException {
    if (finished) {
      throw new IllegalStateException("already finished");
    }
    finished = true;
    if (hnswMeta != null) {
      // write end of fields marker
      hnswMeta.writeInt(-1);
      CodecUtil.writeFooter(hnswMeta);
    }
    if (hnswVectorIndex != null) {
      CodecUtil.writeFooter(hnswVectorIndex);
    }
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(hnswMeta, hnswVectorIndex);
  }
}

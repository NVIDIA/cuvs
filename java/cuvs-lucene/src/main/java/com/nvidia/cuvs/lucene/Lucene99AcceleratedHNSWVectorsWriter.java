/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.createMultiLayerHnswGraph;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.createSingleVectorHnswGraph;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.printInfoStream;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.writeEmpty;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.writeGraph;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.writeMeta;
import static com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsFormat.HNSW_INDEX_CODEC_NAME;
import static com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsFormat.HNSW_INDEX_EXT;
import static com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsFormat.HNSW_META_CODEC_EXT;
import static com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsFormat.HNSW_META_CODEC_NAME;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.closeCuVSResourcesInstance;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.getCuVSResourcesInstance;
import static org.apache.lucene.index.VectorEncoding.FLOAT32;
import static org.apache.lucene.util.RamUsageEstimator.shallowSizeOfInstance;

import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CuVSHostMatrix;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.QuantizationType;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.Sorter.DocMap;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.InfoStream;

/**
 * This class extends upon the KnnVectorsWriter to
 * enable the creation of GPU-based accelerated HNSW based vector search.
 *
 * @since 25.10
 */
public class Lucene99AcceleratedHNSWVectorsWriter extends KnnVectorsWriter {

  private static final long SHALLOW_RAM_BYTES_USED =
      shallowSizeOfInstance(Lucene99AcceleratedHNSWVectorsWriter.class);
  private static final String COMPONENT = "Lucene99AcceleratedHNSWVectorsWriter";
  private static final LuceneProvider LUCENE_PROVIDER;
  private static final Integer VERSION_CURRENT;

  private final AcceleratedHNSWParams acceleratedHNSWParams;
  private final FlatVectorsWriter flatVectorsWriter;
  private final List<FieldWriter> fields = new ArrayList<>();
  private final InfoStream infoStream;
  private IndexOutput hnswMeta = null;
  private IndexOutput hnswVectorIndex = null;
  private String vemFileName;
  private String vexFileName;
  private boolean finished;

  static {
    try {
      LUCENE_PROVIDER = LuceneProvider.getInstance("99");
      VERSION_CURRENT = LUCENE_PROVIDER.getStaticIntParam("VERSION_CURRENT");
    } catch (Exception e) {
      throw new ExceptionInInitializerError(e.getMessage());
    }
  }

  /**
   * Initializes {@link Lucene99AcceleratedHNSWVectorsWriter}
   *
   * @param state instance of the {@link org.apache.lucene.index.SegmentWriteState}
   * @param acceleratedHNSWParams An instance of {@link AcceleratedHNSWParams}
   * @param flatVectorsWriter instance of the {@link org.apache.lucene.codecs.hnsw.FlatVectorsWriter}
   * @throws IOException IOException
   */
  public Lucene99AcceleratedHNSWVectorsWriter(
      SegmentWriteState state,
      AcceleratedHNSWParams acceleratedHNSWParams,
      FlatVectorsWriter flatVectorsWriter)
      throws IOException {
    super();
    this.flatVectorsWriter = flatVectorsWriter;
    this.infoStream = state.infoStream;
    this.acceleratedHNSWParams = acceleratedHNSWParams;
    vemFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, HNSW_META_CODEC_EXT);
    vexFileName =
        IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, HNSW_INDEX_EXT);
    boolean success = false;
    try {
      hnswMeta = state.directory.createOutput(vemFileName, state.context);
      hnswVectorIndex = state.directory.createOutput(vexFileName, state.context);
      CodecUtil.writeIndexHeader(
          hnswMeta,
          HNSW_META_CODEC_NAME,
          VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      CodecUtil.writeIndexHeader(
          hnswVectorIndex,
          HNSW_INDEX_CODEC_NAME,
          VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      success = true;
      printInfoStream(infoStream, COMPONENT, "Lucene99AcceleratedHNSWVectorsWriter is initialized");
    } finally {
      if (success == false) {
        IOUtils.closeWhileHandlingException(this);
      }
    }
  }

  /**
   * Add new field for indexing.
   */
  @Override
  public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
    var encoding = fieldInfo.getVectorEncoding();
    if (encoding != FLOAT32) {
      throw new IllegalArgumentException("Expected float32, got:" + encoding);
    }
    var writer = Objects.requireNonNull(flatVectorsWriter.addField(fieldInfo));
    var cuvsFieldWriter = new FieldWriter(QuantizationType.NONE, fieldInfo, writer);
    fields.add(cuvsFieldWriter);
    return writer;
  }

  /**
   * Flush/sorting path: builds a host matrix from the heap vectors, then delegates to {@link
   * #writeNonTrivialField(FieldInfo, CuVSMatrix)}.
   *
   * @param fieldInfo instance of FieldInfo that has the field description
   * @param vectors vectors to index
   * @throws IOException
   */
  private void writeFieldInternal(FieldInfo fieldInfo, List<float[]> vectors) throws IOException {
    if (writeTrivialField(fieldInfo, vectors.size())) {
      return;
    }
    CuVSMatrix dataset = Utils.createHostFloatMatrix(vectors, fieldInfo.getVectorDimension());
    writeNonTrivialField(fieldInfo, dataset);
  }

  /**
   * Builds the intermediate CAGRA index and writes the HNSW index. This non-trivial path is shared
   * by flushes and merges after zero- and one-vector cases have been handled.
   *
   * @param fieldInfo instance of FieldInfo that has the field description
   * @param dataset matrix of all vectors to index
   * @throws IOException
   */
  private void writeNonTrivialField(FieldInfo fieldInfo, CuVSMatrix dataset) throws IOException {
    try {
      int size = (int) dataset.size();
      CagraIndexParams params =
          CagraIndexParamsFactory.create(acceleratedHNSWParams, dataset.size(), dataset.columns());
      CagraIndex cagraIndex =
          CagraIndex.newBuilder(getCuVSResourcesInstance())
              .withDataset(dataset)
              .withIndexParams(params)
              .build();
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
              QuantizationType.NONE);
      long vectorIndexOffset = hnswVectorIndex.getFilePointer();
      int[][] graphLevelNodeOffsets = writeGraph(hnswGraph, hnswVectorIndex);
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
      cagraIndex.close();
    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
  }

  /** Writes the empty or one-vector representation, if {@code size} is trivial. */
  private boolean writeTrivialField(FieldInfo fieldInfo, int size) throws IOException {
    if (size == 0) {
      writeEmpty(fieldInfo, hnswMeta);
      return true;
    }
    if (size == 1) {
      writeSingleVectorGraph(fieldInfo);
      return true;
    }
    return false;
  }

  /**
   * Build the indexes and writes it to the disk.
   */
  @Override
  public void flush(int maxDoc, DocMap sortMap) throws IOException {
    flatVectorsWriter.flush(maxDoc, sortMap);
    for (var field : fields) {
      if (sortMap == null) {
        writeField(field);
      } else {
        writeSortingField(field, sortMap);
      }
    }
  }

  /**
   * Builds the index and writes it to the disk.
   *
   * @param fieldData
   * @throws IOException
   */
  private void writeField(FieldWriter fieldData) throws IOException {
    writeFieldInternal(fieldData.fieldInfo(), fieldData.getFloatVectors());
  }

  /**
   * Builds the index and writes it to the disk.
   *
   * @param fieldData instance of GPUFieldWriter
   * @param sortMap instance of the DocMap
   * @throws IOException
   */
  private void writeSortingField(FieldWriter fieldData, Sorter.DocMap sortMap) throws IOException {
    DocsWithFieldSet oldDocsWithFieldSet = fieldData.getDocsWithFieldSet();
    final int[] new2OldOrd = new int[oldDocsWithFieldSet.cardinality()];
    mapOldOrdToNewOrd(oldDocsWithFieldSet, sortMap, null, new2OldOrd, null);
    List<float[]> sortedVectors = new ArrayList<float[]>();
    List<float[]> floatVectors = fieldData.getFloatVectors();
    for (int i = 0; i < floatVectors.size(); i++) {
      sortedVectors.add(floatVectors.get(new2OldOrd[i]));
    }
    writeFieldInternal(fieldData.fieldInfo(), sortedVectors);
  }

  /**
   * Builds and writes a single vector graph.
   *
   * @param fieldInfo instance of FieldInfo
   * @throws IOException I/O Exceptions
   */
  private void writeSingleVectorGraph(FieldInfo fieldInfo) throws IOException {
    try {
      int size = 1;
      int dimensions = fieldInfo.getVectorDimension();
      GPUBuiltHnswGraph hnswGraph = createSingleVectorHnswGraph(size, dimensions);
      long vectorIndexOffset = hnswVectorIndex.getFilePointer();
      int[][] graphLevelNodeOffsets = writeGraph(hnswGraph, hnswVectorIndex);
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

  /**
   * Streams merged vectors directly into a native host-memory matrix without materializing a
   * {@code List<float[]>} on the Java heap. This avoids retaining both the heap list and native
   * matrix during a force merge.
   */
  private void vectorBasedMerge(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    try {
      int size = countMergedVectors(fieldInfo, mergeState);
      if (writeTrivialField(fieldInfo, size)) {
        return;
      }
      FloatVectorValues mergedVectors =
          KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
      int dims = fieldInfo.getVectorDimension();
      CuVSMatrix.Builder<CuVSHostMatrix> builder =
          CuVSMatrix.hostBuilder(size, dims, CuVSMatrix.DataType.FLOAT);
      KnnVectorValues.DocIndexIterator it = mergedVectors.iterator();
      for (int doc = it.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = it.nextDoc()) {
        builder.addVector(mergedVectors.vectorValue(it.index()));
      }
      CuVSHostMatrix dataset = builder.build();
      writeNonTrivialField(fieldInfo, dataset);
    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
  }

  /** Counts the live vectors that the merge iterator will actually yield. */
  private static int countMergedVectors(FieldInfo fieldInfo, MergeState mergeState)
      throws IOException {
    FloatVectorValues mergedVectors =
        KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
    int count = 0;
    KnnVectorValues.DocIndexIterator it = mergedVectors.iterator();
    for (int doc = it.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = it.nextDoc()) {
      count++;
    }
    return count;
  }

  /**
   * Write field for merging.
   */
  @Override
  public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    flatVectorsWriter.mergeOneField(fieldInfo, mergeState);
    vectorBasedMerge(fieldInfo, mergeState);
  }

  /**
   * Called once at the end before close.
   */
  @Override
  public void finish() throws IOException {
    if (finished) {
      throw new IllegalStateException("already finished");
    }
    finished = true;
    flatVectorsWriter.finish();
    if (hnswMeta != null) {
      // write end of fields marker
      hnswMeta.writeInt(-1);
      CodecUtil.writeFooter(hnswMeta);
    }
    if (hnswVectorIndex != null) {
      CodecUtil.writeFooter(hnswVectorIndex);
    }
  }

  /**
   * Closes the resources.
   */
  @Override
  public void close() throws IOException {
    printInfoStream(infoStream, COMPONENT, "Closing resources");
    IOUtils.close(hnswMeta, hnswVectorIndex, flatVectorsWriter);
    closeCuVSResourcesInstance();
  }

  /**
   * Returns the memory usage of this object in bytes.
   */
  @Override
  public long ramBytesUsed() {
    long total = SHALLOW_RAM_BYTES_USED;
    for (var field : fields) {
      total += field.ramBytesUsed();
    }
    return total;
  }
}

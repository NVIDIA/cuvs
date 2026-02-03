/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.cagraIndexParams;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.createMultiLayerHnswGraph;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.createSingleVectorHnswGraph;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.printInfoStream;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.quantizeFloatVectorsToBinary;
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
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.apache.lucene.util.RamUsageEstimator.shallowSizeOfInstance;

import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraIndexParams;
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
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.Sorter.DocMap;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.InfoStream;

/**
 * This class extends upon the KnnVectorsWriter to enable the creation of GPU-based accelerated
 * vector search indexes.
 *
 * @since 26.02
 */
public class LuceneAcceleratedHNSWBinaryQuantizedVectorsWriter extends KnnVectorsWriter {

  private static final long SHALLOW_RAM_BYTES_USED =
      shallowSizeOfInstance(LuceneAcceleratedHNSWBinaryQuantizedVectorsWriter.class);
  private static final String COMPONENT = "Lucene99AcceleratedHNSWBinaryQuantizedVectorsWriter";

  private final FlatVectorsWriter flatVectorsWriter;
  private final List<QuantizedFieldWriter> fields = new ArrayList<>();
  private final InfoStream infoStream;
  private final AcceleratedHNSWParams acceleratedHNSWParams;
  private IndexOutput hnswMeta = null, hnswVectorIndex = null;
  private boolean finished;
  private String vemFileName;
  private String vexFileName;

  /**
   * Initializes {@link LuceneAcceleratedHNSWBinaryQuantizedVectorsWriter}
   *
   * @param state instance of the {@link org.apache.lucene.index.SegmentWriteState}
   * @param acceleratedHNSWParams An instance of {@link AcceleratedHNSWParams}
   * @param flatVectorsWriter instance of the {@link org.apache.lucene.codecs.hnsw.FlatVectorsWriter}
   * @throws IOException IOException
   */
  public LuceneAcceleratedHNSWBinaryQuantizedVectorsWriter(
      SegmentWriteState state,
      AcceleratedHNSWParams acceleratedHNSWParams,
      FlatVectorsWriter flatVectorsWriter)
      throws IOException {
    super();
    this.acceleratedHNSWParams = acceleratedHNSWParams;
    this.flatVectorsWriter = flatVectorsWriter;
    this.infoStream = state.infoStream;

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
          Lucene99HnswVectorsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      CodecUtil.writeIndexHeader(
          hnswVectorIndex,
          HNSW_INDEX_CODEC_NAME,
          Lucene99HnswVectorsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);

      success = true;
      printInfoStream(
          infoStream, COMPONENT, "Lucene99AcceleratedHNSWBinaryQuantizedVectorsWriter opened");
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
    VectorEncoding encoding = fieldInfo.getVectorEncoding();
    if (encoding != FLOAT32) {
      throw new IllegalArgumentException("expected float32, got:" + encoding);
    }
    var writer = Objects.requireNonNull(flatVectorsWriter.addField(fieldInfo));
    var cuvsFieldWriter = new QuantizedFieldWriter(QuantizationType.BINARY, fieldInfo, writer);
    fields.add(cuvsFieldWriter);
    return writer;
  }

  /**
   * Builds the intermediate CAGRA index and builds and writes the HNSW index for binary quantized vectors.
   * Binary quantized vectors are stored as packed bits (1 bit per dimension, 8 dimensions per byte).
   *
   * @param fieldInfo instance of FieldInfo that has the field description
   * @param vectors binary quantized vectors (packed bits as bytes)
   * @throws IOException
   */
  private void writeFieldInternal(FieldInfo fieldInfo, List<byte[]> vectors) throws IOException {
    if (vectors.size() == 0) {
      writeEmpty(fieldInfo, hnswMeta);
      return;
    }

    try {
      int dimensions = fieldInfo.getVectorDimension();
      int bytesPerVector = (dimensions + 7) / 8;

      CuVSMatrix dataset =
          Utils.createByteMatrix(vectors, bytesPerVector, getCuVSResourcesInstance());

      if (dataset.size() < 2) {
        writeSingleVectorGraph(fieldInfo, vectors);
        return;
      }

      CagraIndexParams params =
          cagraIndexParams(
              acceleratedHNSWParams.getWriterThreads(),
              acceleratedHNSWParams.getIntermediateGraphDegree(),
              acceleratedHNSWParams.getGraphdegree());
      CagraIndex cagraIndex =
          CagraIndex.newBuilder(getCuVSResourcesInstance())
              .withDataset(dataset)
              .withIndexParams(params)
              .build();

      CuVSMatrix adjacencyListMatrix = cagraIndex.getGraph();
      int size = (int) dataset.size();

      // Create multi-layer HNSW graph from CAGRA
      GPUBuiltHnswGraph hnswGraph =
          createMultiLayerHnswGraph(
              fieldInfo,
              size,
              dimensions,
              adjacencyListMatrix,
              vectors,
              acceleratedHNSWParams.getHnswLayers(),
              acceleratedHNSWParams.getGraphdegree(),
              params,
              QuantizationType.BINARY);

      long vectorIndexOffset = hnswVectorIndex.getFilePointer();
      // Write the graph to the vector index
      int[][] graphLevelNodeOffsets = writeGraph(hnswGraph, hnswVectorIndex);
      long vectorIndexLength = hnswVectorIndex.getFilePointer() - vectorIndexOffset;

      // Write metadata
      writeMeta(
          hnswVectorIndex,
          hnswMeta,
          fieldInfo,
          vectorIndexOffset,
          vectorIndexLength,
          size,
          hnswGraph,
          graphLevelNodeOffsets,
          acceleratedHNSWParams.getGraphdegree());

      cagraIndex.close();

    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
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
  private void writeField(QuantizedFieldWriter fieldData) throws IOException {
    writeFieldInternal(fieldData.fieldInfo(), fieldData.getVectors());
  }

  /**
   * Builds the index and writes it to the disk.
   *
   * @param fieldData instance of BinaryQuantizedGPUFieldWriter
   * @param sortMap instance of the DocMap
   * @throws IOException
   */
  private void writeSortingField(QuantizedFieldWriter fieldData, Sorter.DocMap sortMap)
      throws IOException {

    DocsWithFieldSet oldDocsWithFieldSet = fieldData.getDocsWithFieldSet();
    final int[] new2OldOrd = new int[oldDocsWithFieldSet.cardinality()]; // new ord to old ord
    mapOldOrdToNewOrd(oldDocsWithFieldSet, sortMap, null, new2OldOrd, null);

    List<byte[]> sortedVectors = new ArrayList<byte[]>();
    List<byte[]> vectors = fieldData.getVectors();
    for (int i = 0; i < vectors.size(); i++) {
      sortedVectors.add(vectors.get(new2OldOrd[i]));
    }

    writeFieldInternal(fieldData.fieldInfo(), sortedVectors);
  }

  /**
   * Builds and writes a single vector graph.
   *
   * @param fieldInfo instance of FieldInfo
   * @param vectors the list of binary quantized vectors
   * @throws IOException I/O Exceptions
   */
  private void writeSingleVectorGraph(FieldInfo fieldInfo, List<byte[]> vectors)
      throws IOException {
    // Workaround for CAGRA not supporting single vector indexes
    try {
      int size = 1;
      int dimensions = fieldInfo.getVectorDimension();

      // Create a dummy HNSW graph for a single vector
      GPUBuiltHnswGraph hnswGraph = createSingleVectorHnswGraph(size, dimensions);

      long vectorIndexOffset = hnswVectorIndex.getFilePointer();

      // Write the graph to the vector index
      int[][] graphLevelNodeOffsets = writeGraph(hnswGraph, hnswVectorIndex);

      long vectorIndexLength = hnswVectorIndex.getFilePointer() - vectorIndexOffset;

      // Write metadata
      writeMeta(
          hnswVectorIndex,
          hnswMeta,
          fieldInfo,
          vectorIndexOffset,
          vectorIndexLength,
          size,
          hnswGraph,
          graphLevelNodeOffsets,
          acceleratedHNSWParams.getGraphdegree());

    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
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
   * Fallback method that rebuilds indexes from merged vectors.
   * Used when native CAGRA merge() is not possible.
   */
  private void vectorBasedMerge(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    try {
      FloatVectorValues mergedVectorValues =
          KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);

      if (mergedVectorValues != null) {
        List<float[]> floatVectors = new ArrayList<>();
        KnnVectorValues.DocIndexIterator iter = mergedVectorValues.iterator();
        for (int docV = iter.nextDoc(); docV != NO_MORE_DOCS; docV = iter.nextDoc()) {
          floatVectors.add(mergedVectorValues.vectorValue(iter.index()).clone());
        }
        writeFieldInternal(fieldInfo, quantizeFloatVectorsToBinary(floatVectors));
      }
    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
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

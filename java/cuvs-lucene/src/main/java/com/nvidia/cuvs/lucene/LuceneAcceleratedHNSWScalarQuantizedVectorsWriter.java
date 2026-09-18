/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.apache.lucene.util.RamUsageEstimator.shallowSizeOfInstance;

import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.QuantizationType;
import com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.ScalarQuantizer;
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
public class LuceneAcceleratedHNSWScalarQuantizedVectorsWriter extends KnnVectorsWriter {

  private static final long SHALLOW_RAM_BYTES_USED =
      shallowSizeOfInstance(LuceneAcceleratedHNSWScalarQuantizedVectorsWriter.class);
  private static final String COMPONENT = "Lucene99AcceleratedHNSWQuantizedVectorsWriter";
  private static final LuceneProvider LUCENE_PROVIDER;
  private static final Integer VERSION_CURRENT;

  private final FlatVectorsWriter flatVectorsWriter;
  private final List<FieldWriter> fields = new ArrayList<>();
  private final InfoStream infoStream;
  private final AcceleratedHNSWParams acceleratedHNSWParams;
  private IndexOutput hnswMeta = null, hnswVectorIndex = null;
  private boolean finished;
  private String vemFileName;
  private String vexFileName;

  static {
    try {
      LUCENE_PROVIDER = LuceneProvider.getInstance("99");
      VERSION_CURRENT = LUCENE_PROVIDER.getStaticIntParam("VERSION_CURRENT");
    } catch (Exception e) {
      throw new ExceptionInInitializerError(e.getMessage());
    }
  }

  /**
   * Initializes {@link LuceneAcceleratedHNSWScalarQuantizedVectorsWriter}
   *
   * @param state instance of the {@link org.apache.lucene.index.SegmentWriteState}
   * @param acceleratedHNSWParams An instance of {@link AcceleratedHNSWParams}
   * @param flatVectorsWriter instance of the {@link org.apache.lucene.codecs.hnsw.FlatVectorsWriter}
   * @throws IOException IOException
   */
  public LuceneAcceleratedHNSWScalarQuantizedVectorsWriter(
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
      printInfoStream(
          infoStream, COMPONENT, "Lucene99AcceleratedHNSWQuantizedVectorsWriter opened");
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
    var cuvsFieldWriter = new FieldWriter(QuantizationType.SCALAR, fieldInfo, writer);
    fields.add(cuvsFieldWriter);
    return writer;
  }

  private static byte signedToUnsignedByte(byte signedByte) {
    return (byte) (signedByte & 0xFF);
  }

  /**
   * Builds the intermediate CAGRA index and builds and writes the HNSW index.
   *
   * @param fieldInfo instance of FieldInfo that has the field description
   * @param vectors quantized vectors
   * @throws IOException
   */
  private void writeFieldInternal(FieldInfo fieldInfo, List<?> vectors) throws IOException {
    int size = vectors.size();
    if (writeTrivialField(fieldInfo, size)) {
      return;
    }
    try {
      int dimensions = fieldInfo.getVectorDimension();
      try (CuVSMatrix.Builder<?> builder =
          CuVSMatrix.hostBuilder(size, dimensions, CuVSMatrix.DataType.BYTE)) {
        byte[] unsignedVector = new byte[dimensions];
        for (Object value : vectors) {
          if (!(value instanceof byte[] signedVector) || signedVector.length != dimensions) {
            throw new IllegalArgumentException(
                "Expected scalar-quantized byte[" + dimensions + "] vector");
          }
          copySignedToUnsigned(signedVector, unsignedVector);
          builder.addVector(unsignedVector);
        }
        writeFieldInternal(fieldInfo, builder.build());
      }
    } catch (Throwable t) {
      throw Utils.handleThrowable(t);
    }
  }

  /** Builds and writes an index from an owned scalar-vector matrix. */
  private void writeFieldInternal(FieldInfo fieldInfo, CuVSMatrix dataset) throws IOException {
    try (Utils.OwnedIndex<CagraIndex> ownedIndex = Utils.ownDataset(dataset)) {
      int size = Math.toIntExact(dataset.size());
      if (writeTrivialField(fieldInfo, size)) {
        return;
      }
      int dimensions = fieldInfo.getVectorDimension();
      CagraIndexParams params =
          CagraIndexParamsFactory.create(acceleratedHNSWParams, dataset.size(), dataset.columns());
      CagraIndex cagraIndex =
          CagraIndex.newBuilder(getCuVSResourcesInstance())
              .withDataset(dataset)
              .withIndexParams(params)
              .build();
      ownedIndex.transferTo(cagraIndex);

      CuVSMatrix adjacencyListMatrix = cagraIndex.getGraph();
      GPUBuiltHnswGraph hnswGraph =
          createMultiLayerHnswGraph(
              dimensions,
              adjacencyListMatrix,
              dataset,
              acceleratedHNSWParams.getHnswLayers(),
              params,
              QuantizationType.SCALAR);

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
      throw Utils.handleThrowable(t);
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
    writeFieldInternal(fieldData.fieldInfo(), fieldData.getByteVectors());
  }

  /**
   * Builds the index and writes it to the disk.
   *
   * @param fieldData instance of ScalarQuantizedGPUFieldWriter
   * @param sortMap instance of the DocMap
   * @throws IOException
   */
  private void writeSortingField(FieldWriter fieldData, Sorter.DocMap sortMap) throws IOException {

    DocsWithFieldSet oldDocsWithFieldSet = fieldData.getDocsWithFieldSet();
    final int[] new2OldOrd = new int[oldDocsWithFieldSet.cardinality()]; // new ord to old ord
    mapOldOrdToNewOrd(oldDocsWithFieldSet, sortMap, null, new2OldOrd, null);

    List<byte[]> sortedVectors = new ArrayList<byte[]>();
    List<byte[]> byteVectors = fieldData.getByteVectors();
    for (int i = 0; i < byteVectors.size(); i++) {
      sortedVectors.add(byteVectors.get(new2OldOrd[i]));
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
          graphLevelNodeOffsets);

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
      int dimensions = fieldInfo.getVectorDimension();
      ScalarQuantizer quantizer = collectScalarMergeStats(fieldInfo, mergeState, dimensions);
      if (writeTrivialField(fieldInfo, quantizer.count())) {
        return;
      }

      FloatVectorValues mergedVectors =
          KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
      try (CuVSMatrix.Builder<?> builder =
          CuVSMatrix.hostBuilder(quantizer.count(), dimensions, CuVSMatrix.DataType.BYTE)) {
        byte[] quantized = new byte[dimensions];
        int encodedCount = 0;
        KnnVectorValues.DocIndexIterator iterator = mergedVectors.iterator();
        for (int doc = iterator.nextDoc(); doc != NO_MORE_DOCS; doc = iterator.nextDoc()) {
          if (encodedCount >= quantizer.count()) {
            throw new IOException(
                "Merged vector count changed between quantization passes: expected "
                    + quantizer.count()
                    + ", received more rows");
          }
          quantizer.quantize(mergedVectors.vectorValue(iterator.index()), quantized);
          copySignedToUnsigned(quantized, quantized);
          builder.addVector(quantized);
          encodedCount = Math.incrementExact(encodedCount);
        }
        if (encodedCount != quantizer.count()) {
          throw new IOException(
              "Merged vector count changed between quantization passes: expected "
                  + quantizer.count()
                  + ", received "
                  + encodedCount);
        }
        writeFieldInternal(fieldInfo, builder.build());
      }
    } catch (Throwable t) {
      throw Utils.handleThrowable(t);
    }
  }

  private static ScalarQuantizer collectScalarMergeStats(
      FieldInfo fieldInfo, MergeState mergeState, int dimensions) throws IOException {
    FloatVectorValues mergedVectors =
        KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
    ScalarQuantizer quantizer = new ScalarQuantizer(dimensions);
    if (mergedVectors != null) {
      KnnVectorValues.DocIndexIterator iterator = mergedVectors.iterator();
      for (int doc = iterator.nextDoc(); doc != NO_MORE_DOCS; doc = iterator.nextDoc()) {
        quantizer.add(mergedVectors.vectorValue(iterator.index()));
      }
    }
    if (quantizer.count() > 0) {
      quantizer.finish();
    }
    return quantizer;
  }

  private static void copySignedToUnsigned(byte[] source, byte[] destination) {
    for (int i = 0; i < source.length; i++) {
      destination[i] = signedToUnsignedByte(source[i]);
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

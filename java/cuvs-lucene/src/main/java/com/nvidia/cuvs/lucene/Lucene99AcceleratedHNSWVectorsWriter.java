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
import org.apache.lucene.util.Bits;
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

  /**
   * Hint-path state. When {@code numInputVectors > 0}, vectors are streamed into a native host
   * matrix (see {@link FieldWriter}) rather than a heap {@code List<float[]>}, and the flat
   * {@code .vec}/{@code .vemf} files are written by {@link #nativeFlat} instead of by
   * {@link #flatVectorsWriter} (which is {@code null} in this mode). Supports only the unsorted
   * single-segment flush path; merges and index-sorted flushes are rejected.
   */
  private final int numInputVectors;

  private final boolean nativeMode;
  private final NativeFlatVectorsWriter nativeFlat;
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
    this.numInputVectors = acceleratedHNSWParams.getNumInputVectors();
    this.nativeMode = numInputVectors > 0;
    if (nativeMode && state.segmentInfo.getIndexSort() != null) {
      throw new IllegalArgumentException(
          "AcceleratedHNSWParams.numInputVectors (native flat buffering) does not support"
              + " index-sorted segments; unset it (0) to use the heap-buffered path");
    }
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
      // In hint mode we own the flat files; the Lucene flat writer must be absent to avoid opening
      // the same .vec/.vemf outputs.
      nativeFlat = nativeMode ? new NativeFlatVectorsWriter(state) : null;
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
    if (nativeMode) {
      // Buffer directly into a native host matrix; return the FieldWriter itself so Lucene routes
      // addValue() here rather than to a (nonexistent) Lucene flat field writer.
      var cuvsFieldWriter =
          new FieldWriter(QuantizationType.NONE, fieldInfo, null, numInputVectors);
      fields.add(cuvsFieldWriter);
      return cuvsFieldWriter;
    }
    var writer = Objects.requireNonNull(flatVectorsWriter.addField(fieldInfo));
    var cuvsFieldWriter = new FieldWriter(QuantizationType.NONE, fieldInfo, writer);
    fields.add(cuvsFieldWriter);
    return writer;
  }

  /**
   * Flush/sorting path: builds a host matrix from the heap vectors, then delegates
   * to {@link #writeFieldInternal(FieldInfo, CuVSMatrix)}.
   *
   * @param fieldInfo instance of FieldInfo that has the field description
   * @param vectors vectors to index
   * @throws IOException
   */
  private void writeFieldInternal(FieldInfo fieldInfo, List<float[]> vectors) throws IOException {
    if (vectors.size() == 0) {
      writeEmpty(fieldInfo, hnswMeta);
      return;
    }
    if (vectors.size() < 2) {
      writeSingleVectorGraph(fieldInfo, vectors);
      return;
    }
    CuVSMatrix dataset = Utils.createFloatMatrix(vectors, fieldInfo.getVectorDimension());
    writeFieldInternal(fieldInfo, dataset);
  }

  /**
   * Builds the intermediate CAGRA index and builds and writes the HNSW index.
   * Single implementation used by both the flush and merge paths. The dataset is a
   * {@link CuVSMatrix} (host-backed on the merge path) so the full set of vectors is
   * never double-materialised on the Java heap.
   *
   * @param fieldInfo instance of FieldInfo that has the field description
   * @param dataset   matrix of all vectors to index
   * @throws IOException
   */
  private void writeFieldInternal(FieldInfo fieldInfo, CuVSMatrix dataset) throws IOException {
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
    if (nativeMode) {
      if (sortMap != null) {
        throw new UnsupportedOperationException(
            "AcceleratedHNSWParams.numInputVectors (native flat buffering) does not support"
                + " index-sorted segments; unset it (0) to enable the sorted flush path");
      }
      for (var field : fields) {
        writeFieldNative(field, maxDoc);
      }
      return;
    }
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
   * Hint-path flush for a single field: writes the flat {@code .vec}/{@code .vemf} from the native
   * host matrix, builds the CAGRA/HNSW graph from the same matrix, then releases the matrix. Both
   * consumers read the matrix before it is closed.
   */
  private void writeFieldNative(FieldWriter fieldData, int maxDoc) throws IOException {
    int count = fieldData.getNativeVectorCount();
    if (count != numInputVectors) {
      throw new IllegalStateException(
          "numInputVectors ("
              + numInputVectors
              + ") must equal the number of vectors added ("
              + count
              + ") for field \""
              + fieldData.fieldInfo().name
              + "\"; the native host matrix is sized for the hint exactly. This usually means"
              + " IndexWriterConfig's auto-flush wasn't disabled (setMaxBufferedDocs /"
              + " setRAMBufferSizeMB(DISABLE_AUTO_FLUSH)), so a flush landed before exactly"
              + " numInputVectors vectors were added; see AcceleratedHNSWParams.Builder"
              + "#withNumInputVectors.");
    }
    FieldInfo fieldInfo = fieldData.fieldInfo();
    try {
      CuVSHostMatrix dataset = fieldData.getHostMatrix();
      nativeFlat.writeField(fieldInfo, dataset, maxDoc, fieldData.getDocsWithFieldSet());
      writeFieldInternal(fieldInfo, dataset);
    } finally {
      fieldData.releaseNativeBuffer();
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
   * @param vectors the list of float vectors
   * @throws IOException I/O Exceptions
   */
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

  /**
   * Streams merged vectors directly into a native host-memory matrix (CuVSHostMatrix)
   * without materialising a List<float[]> on the Java heap, then calls writeFieldInternal.
   * This avoids the double-copy OOM (heap list + native matrix simultaneously) that
   * occurs when force-merging large segments.
   */
  private void vectorBasedMerge(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    try {
      // FloatVectorValues#size() on the merged view is the raw sum of every source segment's
      // on-disk vector count (MergedVectorValues.MergedFloat32VectorValues computes it once at
      // construction from each sub-reader's unfiltered size) -- NOT the number of live
      // (non-deleted) vectors the iterator below will actually yield, which is what
      // CuVSMatrix.hostBuilder needs since it preallocates a fixed-size native buffer. Using
      // size() here under-fills that buffer whenever the merge drops deleted docs, leaving the
      // graph built over more rows than were actually populated.
      //
      // size() IS trustworthy when no segment being merged has any deletions: per-segment vector
      // counts already exclude docs without a value for this field (sparse fields are handled at
      // the single-segment level, independent of deletions), so the raw sum equals the live count
      // in that case and the extra counting pass below can be skipped.
      boolean anySegmentHasDeletions = false;
      for (Bits liveDocs : mergeState.liveDocs) {
        if (liveDocs != null) {
          anySegmentHasDeletions = true;
          break;
        }
      }

      int size;
      if (anySegmentHasDeletions) {
        // Count the live vectors via a throwaway iteration first (mergeFloatVectorValues
        // constructs a fresh, independent view each call, so this doesn't disturb the real build
        // pass below).
        size = 0;
        FloatVectorValues counting =
            KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
        KnnVectorValues.DocIndexIterator countingIt = counting.iterator();
        for (int doc = countingIt.nextDoc();
            doc != DocIdSetIterator.NO_MORE_DOCS;
            doc = countingIt.nextDoc()) {
          size++;
        }
      } else {
        size =
            KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState)
                .size();
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
      writeFieldInternal(fieldInfo, dataset);
    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
  }

  /**
   * Write field for merging.
   */
  @Override
  public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    if (nativeMode) {
      throw new UnsupportedOperationException(
          "AcceleratedHNSWParams.numInputVectors (native flat buffering) supports only the"
              + " unsorted single-segment flush path; unset it (0) to enable merges");
    }
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
    if (nativeMode) {
      nativeFlat.finish();
    } else {
      flatVectorsWriter.finish();
    }
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
    IOUtils.close(hnswMeta, hnswVectorIndex, flatVectorsWriter, nativeFlat);
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

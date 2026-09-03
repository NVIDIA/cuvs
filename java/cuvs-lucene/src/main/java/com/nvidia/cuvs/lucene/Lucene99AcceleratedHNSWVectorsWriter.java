/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.printInfoStream;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.closeCuVSResourcesInstance;
import static org.apache.lucene.index.VectorEncoding.FLOAT32;
import static org.apache.lucene.util.RamUsageEstimator.shallowSizeOfInstance;

import com.nvidia.cuvs.CuVSHostMatrix;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.QuantizationType;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.Sorter.DocMap;
import org.apache.lucene.search.DocIdSetIterator;
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

  private final FlatVectorsWriter flatVectorsWriter;
  private final List<FieldWriter> fields = new ArrayList<>();
  private final InfoStream infoStream;
  private AcceleratedHnswGraphOutput graphOutput;
  private boolean finished;

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
    this.flatVectorsWriter = Objects.requireNonNull(flatVectorsWriter);
    this.infoStream = state.infoStream;
    boolean success = false;
    try {
      graphOutput = new AcceleratedHnswGraphOutput(state, acceleratedHNSWParams);
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
    graphOutput.writeField(fieldData.fieldInfo(), fieldData.getFloatVectors());
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
    graphOutput.writeField(fieldData.fieldInfo(), sortedVectors);
  }

  /**
   * Streams merged vectors directly into a native host-memory matrix (CuVSHostMatrix)
   * without materialising a List<float[]> on the Java heap, then calls graphOutput.writeField.
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
      graphOutput.writeField(fieldInfo, dataset);
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
   * Called once at the end before close.
   */
  @Override
  public void finish() throws IOException {
    if (finished) {
      throw new IllegalStateException("already finished");
    }
    finished = true;
    flatVectorsWriter.finish();
    graphOutput.finish();
  }

  /**
   * Closes the resources.
   */
  @Override
  public void close() throws IOException {
    printInfoStream(infoStream, COMPONENT, "Closing resources");
    IOUtils.close(graphOutput, flatVectorsWriter);
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

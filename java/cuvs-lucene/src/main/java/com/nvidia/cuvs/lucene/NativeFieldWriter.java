/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CuVSHostMatrix;
import com.nvidia.cuvs.CuVSMatrix;
import java.io.IOException;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.util.RamUsageEstimator;

/**
 * Native-flat-buffered field writer, used exclusively by {@link
 * NativeFlatBufferedHNSWVectorsWriter}: streams incoming vectors directly into a native host
 * matrix (see {@link CuVSMatrix#hostBuilder}) instead of accumulating a {@code List<float[]>} on
 * the Java heap, so the full dataset is never held twice (heap list + native copy). The matrix is
 * preallocated for exactly {@code numInputVectors} rows, so that count must equal the number of
 * vectors actually added (validated by {@link NativeFlatBufferedHNSWVectorsWriter}).
 *
 * <p>Package-private and float32-only — quantization is not supported on this path. Distinct from
 * the heap-buffered {@link FieldWriter} used by the other accelerated writers; the two share no
 * state or behavior beyond both extending {@link KnnFieldVectorsWriter}.
 */
final class NativeFieldWriter extends KnnFieldVectorsWriter<Object> {

  private static final long SHALLOW_SIZE =
      RamUsageEstimator.shallowSizeOfInstance(NativeFieldWriter.class);

  private final FieldInfo fieldInfo;
  private final int numInputVectors;
  private final CuVSMatrix.Builder<CuVSHostMatrix> hostMatrixBuilder;
  private final DocsWithFieldSet docsWithField = new DocsWithFieldSet();
  private int lastDocID = -1;
  private int count;
  private CuVSHostMatrix builtMatrix;

  NativeFieldWriter(FieldInfo fieldInfo, int numInputVectors) {
    this.fieldInfo = fieldInfo;
    this.numInputVectors = numInputVectors;
    // Preallocates one contiguous native region of numInputVectors * dimension * 4 bytes.
    this.hostMatrixBuilder =
        CuVSMatrix.hostBuilder(
            numInputVectors, fieldInfo.getVectorDimension(), CuVSMatrix.DataType.FLOAT);
  }

  @Override
  public void addValue(int docID, Object vectorValue) throws IOException {
    if (docID == lastDocID) {
      throw new IllegalArgumentException(
          "VectorValuesField \""
              + fieldInfo.name
              + "\" appears more than once in this document (only one value is allowed per"
              + " field)");
    }
    if (count >= numInputVectors) {
      throw new IllegalStateException(
          "Buffered vectors ("
              + (count + 1)
              + ") exceed numInputVectors ("
              + numInputVectors
              + ") for field \""
              + fieldInfo.name
              + "\". This usually means more vectors arrived than the numInputVectors hint"
              + " promised (e.g. a merge or a later flush cycle reused this config); see"
              + " CagraHnswBulkIndexWriter.");
    }
    // hostMatrixBuilder.addVector validates the dimension and performs the native row copy.
    hostMatrixBuilder.addVector((float[]) vectorValue);
    docsWithField.add(docID);
    count++;
    lastDocID = docID;
  }

  FieldInfo fieldInfo() {
    return fieldInfo;
  }

  DocsWithFieldSet getDocsWithFieldSet() {
    return docsWithField;
  }

  /**
   * The native host matrix holding the buffered vectors. Built once and cached; the caller owns
   * closing it via {@link #releaseNativeBuffer()} once the CAGRA build has consumed it.
   */
  CuVSHostMatrix getHostMatrix() {
    if (builtMatrix == null) {
      builtMatrix = hostMatrixBuilder.build();
    }
    return builtMatrix;
  }

  /** Number of vectors buffered so far. */
  int getNativeVectorCount() {
    return count;
  }

  /** Closes the native host matrix. Safe to call multiple times. */
  void releaseNativeBuffer() {
    getHostMatrix().close();
    builtMatrix = null;
  }

  @Override
  public Object copyValue(Object vectorValue) {
    throw new UnsupportedOperationException();
  }

  @Override
  public long ramBytesUsed() {
    // The native host matrix is off-heap and intentionally excluded from Lucene's heap RAM
    // accounting.
    return SHALLOW_SIZE;
  }
}

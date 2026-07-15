/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.quantizeFloatVectorsToBinary;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.quantizeFloatVectorsToScalar;

import com.nvidia.cuvs.CuVSHostMatrix;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.QuantizationType;
import java.io.IOException;
import java.util.List;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.util.RamUsageEstimator;

public class FieldWriter extends KnnFieldVectorsWriter<Object> {

  private static final long SHALLOW_SIZE =
      RamUsageEstimator.shallowSizeOfInstance(FieldWriter.class);

  private final FieldInfo fieldInfo;
  private final int dimension;
  private final FlatFieldVectorsWriter<float[]> flatFieldVectorsWriter;
  private int lastDocID = -1;
  private QuantizationType quantizationType;

  /**
   * Native-buffering state. When {@code numInputVectors > 0} the writer streams incoming vectors
   * directly into a native host matrix ({@link CuVSMatrix#hostBuilder}) instead of accumulating a
   * {@code List<float[]>} on the Java heap via {@link #flatFieldVectorsWriter}. That matrix is reused
   * as the CAGRA build input, so the full dataset is never held twice (heap list + native copy).
   *
   * <p>The matrix is preallocated for exactly {@code numInputVectors} rows, so the hint must equal
   * the number of vectors actually added (validated at build time in the caller).
   */
  private final int numInputVectors;

  private final boolean nativeBuffering;
  private final CuVSMatrix.Builder<CuVSHostMatrix> hostMatrixBuilder;
  private final DocsWithFieldSet nativeDocsWithField;
  private int nativeCount;
  private CuVSHostMatrix builtMatrix;

  @SuppressWarnings("unchecked")
  public FieldWriter(
      QuantizationType quantizationType,
      FieldInfo fieldInfo,
      FlatFieldVectorsWriter<?> flatFieldVectorsWriter) {
    this(quantizationType, fieldInfo, flatFieldVectorsWriter, 0);
  }

  @SuppressWarnings("unchecked")
  public FieldWriter(
      QuantizationType quantizationType,
      FieldInfo fieldInfo,
      FlatFieldVectorsWriter<?> flatFieldVectorsWriter,
      int numInputVectors) {
    this.quantizationType = quantizationType;
    this.fieldInfo = fieldInfo;
    this.dimension = fieldInfo.getVectorDimension();
    this.flatFieldVectorsWriter = (FlatFieldVectorsWriter<float[]>) flatFieldVectorsWriter;
    this.numInputVectors = numInputVectors;
    this.nativeBuffering = numInputVectors > 0;
    if (nativeBuffering) {
      // Preallocates one contiguous native region of numInputVectors * dimension * 4 bytes.
      this.hostMatrixBuilder =
          CuVSMatrix.hostBuilder(numInputVectors, dimension, CuVSMatrix.DataType.FLOAT);
      this.nativeDocsWithField = new DocsWithFieldSet();
    } else {
      this.hostMatrixBuilder = null;
      this.nativeDocsWithField = null;
    }
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
    if (nativeBuffering) {
      if (nativeCount >= numInputVectors) {
        throw new IllegalStateException(
            "Buffered vectors ("
                + (nativeCount + 1)
                + ") exceed numInputVectors ("
                + numInputVectors
                + ") for field \""
                + fieldInfo.name
                + "\"");
      }
      // hostMatrixBuilder.addVector validates the dimension and performs the native row copy.
      hostMatrixBuilder.addVector((float[]) vectorValue);
      nativeDocsWithField.add(docID);
      nativeCount++;
      lastDocID = docID;
    } else {
      flatFieldVectorsWriter.addValue(docID, (float[]) vectorValue);
    }
  }

  List<byte[]> getByteVectors() {
    if (quantizationType == QuantizationType.BINARY) {
      return quantizeFloatVectorsToBinary(flatFieldVectorsWriter.getVectors());
    } else if (quantizationType == QuantizationType.SCALAR) {
      return quantizeFloatVectorsToScalar(flatFieldVectorsWriter.getVectors());
    } else {
      throw new UnsupportedOperationException("Not applicable for QuantizationType.NONE");
    }
  }

  List<float[]> getFloatVectors() {
    return flatFieldVectorsWriter.getVectors();
  }

  FieldInfo fieldInfo() {
    return fieldInfo;
  }

  DocsWithFieldSet getDocsWithFieldSet() {
    return nativeBuffering ? nativeDocsWithField : flatFieldVectorsWriter.getDocsWithFieldSet();
  }

  /** Whether this writer streams vectors into a native host matrix (hint path). */
  boolean isNativeBuffering() {
    return nativeBuffering;
  }

  /**
   * The native host matrix holding the buffered vectors. Valid only when {@link #isNativeBuffering()}
   * is true. The matrix is built once and cached; the caller owns closing it via
   * {@link #releaseNativeBuffer()} once the CAGRA build has consumed it.
   */
  CuVSHostMatrix getHostMatrix() {
    if (builtMatrix == null) {
      builtMatrix = hostMatrixBuilder.build();
    }
    return builtMatrix;
  }

  /** Number of vectors buffered so far (hint path). */
  int getNativeVectorCount() {
    return nativeCount;
  }

  int dimension() {
    return dimension;
  }

  /** Closes the native host matrix. Safe to call multiple times; a no-op in the non-native path. */
  void releaseNativeBuffer() {
    if (nativeBuffering) {
      getHostMatrix().close();
      builtMatrix = null;
    }
  }

  @Override
  public Object copyValue(Object vectorValue) {
    throw new UnsupportedOperationException();
  }

  @Override
  public long ramBytesUsed() {
    // The native host matrix is off-heap and intentionally excluded from Lucene's heap RAM accounting.
    return SHALLOW_SIZE + (nativeBuffering ? 0 : flatFieldVectorsWriter.ramBytesUsed());
  }
}

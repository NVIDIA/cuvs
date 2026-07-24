/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.quantizeFloatVectorsToBinary;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.quantizeFloatVectorsToScalar;

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
  private final FlatFieldVectorsWriter<float[]> flatFieldVectorsWriter;
  private int lastDocID = -1;
  private QuantizationType quantizationType;

  @SuppressWarnings("unchecked")
  public FieldWriter(
      QuantizationType quantizationType,
      FieldInfo fieldInfo,
      FlatFieldVectorsWriter<?> flatFieldVectorsWriter) {
    this.quantizationType = quantizationType;
    this.fieldInfo = fieldInfo;
    this.flatFieldVectorsWriter = (FlatFieldVectorsWriter<float[]>) flatFieldVectorsWriter;
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
    flatFieldVectorsWriter.addValue(docID, (float[]) vectorValue);
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
    return flatFieldVectorsWriter.getDocsWithFieldSet();
  }

  @Override
  public Object copyValue(Object vectorValue) {
    throw new UnsupportedOperationException();
  }

  @Override
  public long ramBytesUsed() {
    return SHALLOW_SIZE + flatFieldVectorsWriter.ramBytesUsed();
  }
}

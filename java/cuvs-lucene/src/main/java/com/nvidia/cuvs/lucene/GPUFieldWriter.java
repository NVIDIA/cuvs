/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.util.List;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.util.RamUsageEstimator;

/**
 * cuVS based fields writer
 *
 * @since 25.10
 */
/*package-private*/ class GPUFieldWriter extends KnnFieldVectorsWriter<float[]> {

  private static final long SHALLOW_SIZE =
      RamUsageEstimator.shallowSizeOfInstance(GPUFieldWriter.class);

  private final FieldInfo fieldInfo;
  private final FlatFieldVectorsWriter<float[]> flatFieldVectorsWriter;
  private int lastDocID = -1;

  public GPUFieldWriter(
      FieldInfo fieldInfo, FlatFieldVectorsWriter<float[]> flatFieldVectorsWriter) {
    this.fieldInfo = fieldInfo;
    this.flatFieldVectorsWriter = flatFieldVectorsWriter;
  }

  /**
   * Add new docID with its vector value to the given field for indexing.
   */
  @Override
  public void addValue(int docID, float[] vectorValue) throws IOException {
    if (docID == lastDocID) {
      throw new IllegalArgumentException(
          "VectorValuesField \""
              + fieldInfo.name
              + "\" appears more than once in this document (only one value is allowed per field)");
    }
    flatFieldVectorsWriter.addValue(docID, vectorValue);
  }

  /**
   * Gets the list of float vectors.
   *
   * @return a list of float vectors
   */
  List<float[]> getVectors() {
    return flatFieldVectorsWriter.getVectors();
  }

  /**
   * Gets the field info that holds the description of the field.
   *
   * @return an instance of FieldInfo
   */
  FieldInfo fieldInfo() {
    return fieldInfo;
  }

  /**
   * Gets the docsWithFieldSet for the field writer.
   *
   * @return an instance of DocsWithFieldSet
   */
  DocsWithFieldSet getDocsWithFieldSet() {
    return flatFieldVectorsWriter.getDocsWithFieldSet();
  }

  /**
   * Used to copy values being indexed to internal storage.
   */
  @Override
  public float[] copyValue(float[] vectorValue) {
    throw new UnsupportedOperationException();
  }

  /**
   * Returns the memory usage of this object in bytes.
   */
  @Override
  public long ramBytesUsed() {
    return SHALLOW_SIZE + flatFieldVectorsWriter.ramBytesUsed();
  }

  /**
   * Returns a string containing the field name and number.
   */
  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder(this.getClass().getSimpleName());
    sb.append("(field name=").append(fieldInfo.name);
    sb.append("number=").append(fieldInfo.number);
    sb.append(")");
    return sb.toString();
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.util.RamUsageEstimator;

/** Validates dense row-ordered placeholders while vectors remain in a borrowed native dataset. */
final class BorrowedDatasetFieldWriter extends KnnFieldVectorsWriter<Object> {

  private static final long SHALLOW_SIZE =
      RamUsageEstimator.shallowSizeOfInstance(BorrowedDatasetFieldWriter.class);

  private final FieldInfo fieldInfo;
  private final int expectedCount;
  private final DocsWithFieldSet docsWithField = new DocsWithFieldSet();
  private int count;

  BorrowedDatasetFieldWriter(FieldInfo fieldInfo, int expectedCount) {
    if (fieldInfo.getVectorEncoding() != VectorEncoding.FLOAT32) {
      throw new IllegalArgumentException("Borrowed FBIN datasets require FLOAT32 vectors");
    }
    this.fieldInfo = fieldInfo;
    this.expectedCount = expectedCount;
  }

  @Override
  public void addValue(int docID, Object vectorValue) throws IOException {
    if (docID != count) {
      throw new IllegalArgumentException(
          "Borrowed FBIN datasets require dense document IDs in row order; expected "
              + count
              + " but got "
              + docID);
    }
    if (!(vectorValue instanceof float[] vector)
        || vector.length != fieldInfo.getVectorDimension()) {
      throw new IllegalArgumentException(
          "Expected a float vector with dimension "
              + fieldInfo.getVectorDimension()
              + " for field "
              + fieldInfo.name);
    }
    if (count >= expectedCount) {
      throw new IllegalStateException(
          "Borrowed FBIN vector count exceeds expected count " + expectedCount);
    }
    docsWithField.add(docID);
    count++;
  }

  FieldInfo fieldInfo() {
    return fieldInfo;
  }

  DocsWithFieldSet docsWithField() {
    return docsWithField;
  }

  int count() {
    return count;
  }

  @Override
  public Object copyValue(Object vectorValue) {
    throw new UnsupportedOperationException();
  }

  @Override
  public long ramBytesUsed() {
    return SHALLOW_SIZE;
  }
}

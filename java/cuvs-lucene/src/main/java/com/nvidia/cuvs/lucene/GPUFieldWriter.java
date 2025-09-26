/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.util.List;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.util.RamUsageEstimator;

/** CuVS based fields writer */
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

  List<float[]> getVectors() {
    return flatFieldVectorsWriter.getVectors();
  }

  FieldInfo fieldInfo() {
    return fieldInfo;
  }

  DocsWithFieldSet getDocsWithFieldSet() {
    return flatFieldVectorsWriter.getDocsWithFieldSet();
  }

  @Override
  public float[] copyValue(float[] vectorValue) {
    throw new UnsupportedOperationException();
  }

  @Override
  public long ramBytesUsed() {
    return SHALLOW_SIZE + flatFieldVectorsWriter.ramBytesUsed();
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder(this.getClass().getSimpleName());
    sb.append("(field name=").append(fieldInfo.name);
    sb.append("number=").append(fieldInfo.number);
    sb.append(")");
    return sb.toString();
  }
}

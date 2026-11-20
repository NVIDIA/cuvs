/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.util.Map;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.DataAccessHint;

/**
 * The part of {@link FilterKnnVectorsReader} whose signatures differ between Lucene releases: since
 * Lucene 10.3, the documents a search may return come as {@link AcceptDocs}, where Lucene 10.2
 * passed {@code Bits}.
 *
 * <p>This variant also forwards {@code getOffHeapByteSize}, which Lucene 10.3 added, and the vector
 * counts and the reads with a {@link DataAccessHint} that Lucene 10.6 added.
 */
abstract class CompatFilterKnnVectorsReader extends FilterKnnVectorsReader {

  /**
   * @param in the reader to forward to
   */
  CompatFilterKnnVectorsReader(KnnVectorsReader in) {
    super(in);
  }

  @Override
  public FloatVectorValues getFloatVectorValues(String field, DataAccessHint access)
      throws IOException {
    return in.getFloatVectorValues(field, access);
  }

  @Override
  public int getVectorCount(FieldInfo fieldInfo) throws IOException {
    return in.getVectorCount(fieldInfo);
  }

  @Override
  public void search(String field, float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs)
      throws IOException {
    in.search(field, target, knnCollector, acceptDocs);
  }

  @Override
  public void search(String field, byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs)
      throws IOException {
    in.search(field, target, knnCollector, acceptDocs);
  }

  @Override
  public Map<String, Long> getOffHeapByteSize(FieldInfo fieldInfo) {
    return in.getOffHeapByteSize(fieldInfo);
  }
}

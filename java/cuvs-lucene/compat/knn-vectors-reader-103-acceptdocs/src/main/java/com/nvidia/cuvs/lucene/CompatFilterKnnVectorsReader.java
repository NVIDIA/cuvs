/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.util.Map;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;

/**
 * Base class of the readers that forward to another {@link KnnVectorsReader}, such as the view of a
 * source segment of a merge in {@code FallbackHnswVectorsWriter}. It forwards the methods of its Lucene
 * release, whose signatures differ between releases: since Lucene 10.3, the documents a search may
 * return come as {@link AcceptDocs}, where Lucene 10.2 passed {@code Bits}.
 *
 * <p>This variant also forwards {@code getOffHeapByteSize}, which Lucene 10.3 added. It does not
 * close the reader it forwards to, which it does not own, and leaves {@code getMergeInstance} and
 * {@code finishMerge} to Lucene's defaults: its subclasses only stand for a source segment inside a
 * merge, whose reader is already the merge instance.
 */
abstract class CompatFilterKnnVectorsReader extends KnnVectorsReader {

  /** The reader this one forwards to. */
  final KnnVectorsReader in;

  /**
   * @param in the reader to forward to
   */
  CompatFilterKnnVectorsReader(KnnVectorsReader in) {
    this.in = in;
  }

  @Override
  public void checkIntegrity() throws IOException {
    in.checkIntegrity();
  }

  @Override
  public FloatVectorValues getFloatVectorValues(String field) throws IOException {
    return in.getFloatVectorValues(field);
  }

  @Override
  public ByteVectorValues getByteVectorValues(String field) throws IOException {
    return in.getByteVectorValues(field);
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

  @Override
  public void close() {}
}

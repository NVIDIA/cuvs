/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.util.Map;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.util.Bits;

/**
 * Base class of the GPU search reader, {@code CuVS2510GPUVectorsReader}, that hides a signature
 * change in {@link KnnVectorsReader}. Since Lucene 10.3, the documents a search may return come as
 * {@link AcceptDocs}, where Lucene 10.2 passed {@code Bits}. Each variant of this class overrides
 * the {@code search} methods of its Lucene release and forwards them to {@code doSearch}, whose
 * signature is the same on every release, so the shared reader implements {@code doSearch} once.
 *
 * <p>This variant converts with {@code AcceptDocs.bits()}. It also reports the reader's off-heap
 * size, which Lucene 10.3 started asking for.
 */
abstract class CompatKnnVectorsReader extends KnnVectorsReader {

  private final FlatVectorsReader rawVectorsReader;

  /**
   * @param rawVectorsReader reads the raw vectors of the fields, which are all of the reader's
   *     off-heap data: the cuVS indexes are loaded into GPU memory
   */
  CompatKnnVectorsReader(FlatVectorsReader rawVectorsReader) {
    this.rawVectorsReader = rawVectorsReader;
  }

  @Override
  public Map<String, Long> getOffHeapByteSize(FieldInfo fieldInfo) {
    return rawVectorsReader.getOffHeapByteSize(fieldInfo);
  }

  @Override
  public void search(String field, float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs)
      throws IOException {
    doSearch(field, target, knnCollector, acceptDocs == null ? null : acceptDocs.bits());
  }

  @Override
  public void search(String field, byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs)
      throws IOException {
    doSearch(field, target, knnCollector, acceptDocs == null ? null : acceptDocs.bits());
  }

  /**
   * Implements {@link #search(String, float[], KnnCollector, AcceptDocs)}.
   *
   * @param acceptDocs the documents that may be returned, or {@code null} if all may be
   */
  protected abstract void doSearch(
      String field, float[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException;

  /**
   * Implements {@link #search(String, byte[], KnnCollector, AcceptDocs)}.
   *
   * @param acceptDocs the documents that may be returned, or {@code null} if all may be
   */
  protected abstract void doSearch(
      String field, byte[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException;
}

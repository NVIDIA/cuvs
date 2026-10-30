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
 * size, which Lucene 10.3 started asking for, and lets the raw vectors reader prepare for a merge,
 * whose methods throw {@link IOException} since Lucene 10.3.
 */
abstract class CompatKnnVectorsReader extends KnnVectorsReader {

  private final FlatVectorsReader rawVectorsReader;

  /**
   * @param rawVectorsReader reads the raw vectors of the fields
   */
  CompatKnnVectorsReader(FlatVectorsReader rawVectorsReader) {
    this.rawVectorsReader = rawVectorsReader;
  }

  /**
   * Reports only the raw vectors. Lucene asks for the memory that searches need to avoid page
   * faults; searches read the raw vectors from their file, but search the cuVS indexes on the GPU.
   * The file holding the cuVS indexes is only read to load them onto the GPU, so it is left out,
   * even though it stays open.
   */
  @Override
  public Map<String, Long> getOffHeapByteSize(FieldInfo fieldInfo) {
    return rawVectorsReader.getOffHeapByteSize(fieldInfo);
  }

  /**
   * Has the raw vectors reader read sequentially, as merges read every vector in order, and
   * returns this reader.
   */
  @Override
  public KnnVectorsReader getMergeInstance() throws IOException {
    // The subclass reads the raw vectors through the reader it was created with, so this relies on
    // the raw vectors reader preparing itself for the merge and returning itself, as
    // Lucene99FlatVectorsReader does, rather than returning another instance.
    rawVectorsReader.getMergeInstance();
    return this;
  }

  /** Has the raw vectors reader go back to the read pattern of searches. */
  @Override
  public void finishMerge() throws IOException {
    rawVectorsReader.finishMerge();
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

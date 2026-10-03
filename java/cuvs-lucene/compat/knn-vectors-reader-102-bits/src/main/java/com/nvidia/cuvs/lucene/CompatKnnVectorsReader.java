/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.util.Bits;

/**
 * A {@link KnnVectorsReader} whose search methods have the same signature on every Lucene release.
 * Lucene 10.2 passes the accepted documents as {@link Bits}.
 */
abstract class CompatKnnVectorsReader extends KnnVectorsReader {

  /**
   * @param rawVectorsReader reads the raw vectors of the fields; unused on Lucene 10.2, which does
   *     not ask for the off-heap size
   */
  CompatKnnVectorsReader(FlatVectorsReader rawVectorsReader) {}

  @Override
  public void search(String field, float[] target, KnnCollector knnCollector, Bits acceptDocs)
      throws IOException {
    doSearch(field, target, knnCollector, acceptDocs);
  }

  @Override
  public void search(String field, byte[] target, KnnCollector knnCollector, Bits acceptDocs)
      throws IOException {
    doSearch(field, target, knnCollector, acceptDocs);
  }

  /**
   * Implements {@link #search(String, float[], KnnCollector, Bits)}.
   *
   * @param acceptDocs the documents that may be returned, or {@code null} if all may be
   */
  protected abstract void doSearch(
      String field, float[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException;

  /**
   * Implements {@link #search(String, byte[], KnnCollector, Bits)}.
   *
   * @param acceptDocs the documents that may be returned, or {@code null} if all may be
   */
  protected abstract void doSearch(
      String field, byte[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException;
}

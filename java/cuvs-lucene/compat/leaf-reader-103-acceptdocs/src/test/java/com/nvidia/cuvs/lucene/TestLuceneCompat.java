/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.util.Bits;

/** The Lucene test calls that differ between the Lucene releases cuvs-lucene supports. */
final class TestLuceneCompat {

  private TestLuceneCompat() {}

  /**
   * Calls {@link LeafReader#searchNearestVectors} with the accepted documents given as {@link
   * Bits}. Since Lucene 10.3 it takes them as {@link AcceptDocs}.
   */
  static void searchNearestVectors(
      LeafReader reader, String field, float[] target, KnnCollector collector, Bits acceptDocs)
      throws IOException {
    reader.searchNearestVectors(
        field, target, collector, AcceptDocs.fromLiveDocs(acceptDocs, reader.maxDoc()));
  }

  /**
   * Calls {@link LeafReader#searchNearestVectors} with the accepted documents given as {@link
   * Bits}, {@code null} meaning all. Since Lucene 10.3 it takes them as {@link AcceptDocs}, which
   * must not be {@code null}.
   */
  static TopDocs searchNearestVectors(
      LeafReader reader, String field, float[] target, int k, Bits acceptDocs, int visitedLimit)
      throws IOException {
    return reader.searchNearestVectors(
        field, target, k, AcceptDocs.fromLiveDocs(acceptDocs, reader.maxDoc()), visitedLimit);
  }
}

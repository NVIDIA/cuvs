/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.util.Bits;

/** The Lucene test calls that differ between the Lucene releases cuvs-lucene supports. */
final class TestLuceneCompat {

  private TestLuceneCompat() {}

  /**
   * Calls {@link LeafReader#searchNearestVectors} with the accepted documents given as {@link
   * Bits}, which is what Lucene 10.2 takes.
   */
  static void searchNearestVectors(
      LeafReader reader, String field, float[] target, KnnCollector collector, Bits acceptDocs)
      throws IOException {
    reader.searchNearestVectors(field, target, collector, acceptDocs);
  }

  /**
   * Calls {@link LeafReader#searchNearestVectors} with the accepted documents given as {@link
   * Bits}, {@code null} meaning all, which is what Lucene 10.2 takes.
   */
  static TopDocs searchNearestVectors(
      LeafReader reader, String field, float[] target, int k, Bits acceptDocs, int visitedLimit)
      throws IOException {
    return reader.searchNearestVectors(field, target, k, acceptDocs, visitedLimit);
  }
}

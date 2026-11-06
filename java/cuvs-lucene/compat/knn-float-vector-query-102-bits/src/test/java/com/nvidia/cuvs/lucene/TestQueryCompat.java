/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.util.Bits;

/** The test calls to {@link CompatKnnFloatVectorQuery} that differ between Lucene releases. */
final class TestQueryCompat {

  private TestQueryCompat() {}

  /**
   * Calls {@code approximateSearch} of a query with the accepted documents given as {@link Bits},
   * which is what Lucene 10.2 takes.
   */
  static TopDocs approximateSearch(
      CompatKnnFloatVectorQuery query, LeafReaderContext context, Bits acceptDocs, int visitedLimit)
      throws IOException {
    return query.approximateSearch(context, acceptDocs, visitedLimit, null);
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.TopKnnCollectorManager;

/**
 * A {@link KnnFloatVectorQuery} whose per-segment search has the same signature on every Lucene
 * release. Since Lucene 10.3, the accepted documents come as {@link AcceptDocs}.
 */
abstract class CompatKnnFloatVectorQuery extends KnnFloatVectorQuery {

  CompatKnnFloatVectorQuery(String field, float[] target, int k, Query filter) {
    super(field, target, k, filter);
    LuceneVersionGuard.ensureCompatible();
  }

  /** Returns the collector for searching one segment, given its visit budget. */
  abstract KnnCollector newPerLeafCollector(int visitedLimit);

  /**
   * Returns a manager that is not optimistic. Since Lucene 10.3, {@code rewrite} searches again,
   * expecting more hits, every segment whose hits all beat the global top-k when the manager is
   * optimistic. {@link #approximateSearch} does not use the manager, so that second search would
   * only repeat the first one.
   */
  @Override
  protected KnnCollectorManager getKnnCollectorManager(int k, IndexSearcher searcher) {
    // A method reference does not override isOptimistic(), which defaults to false.
    return new TopKnnCollectorManager(k, searcher)::newCollector;
  }

  @Override
  protected TopDocs approximateSearch(
      LeafReaderContext context,
      AcceptDocs acceptDocs,
      int visitedLimit,
      KnnCollectorManager knnCollectorManager)
      throws IOException {
    KnnCollector results = newPerLeafCollector(visitedLimit);
    context.reader().searchNearestVectors(field, getTargetCopy(), results, acceptDocs);
    return results.topDocs();
  }
}

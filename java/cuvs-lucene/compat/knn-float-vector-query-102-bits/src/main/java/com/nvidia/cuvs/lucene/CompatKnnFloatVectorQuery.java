/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.util.Bits;

/**
 * Base class of the GPU query, {@code GPUKnnFloatVectorQuery}, that hides a signature change in
 * {@link KnnFloatVectorQuery}. Lucene 10.2 passes the documents a segment search may return to
 * {@code approximateSearch} as {@link Bits}; Lucene 10.3 changed that parameter to
 * {@code AcceptDocs}. Each variant of this class implements the {@code approximateSearch} of its
 * Lucene release with the collector from {@code newPerLeafCollector}, so the shared query only
 * provides that collector.
 *
 * <p>Each variant also tells the shared query how many hits Lucene expects from a segment search,
 * {@link #perLeafTopK}: Lucene 10.2 expects {@code k}, Lucene 10.3 more.
 */
abstract class CompatKnnFloatVectorQuery extends KnnFloatVectorQuery {

  CompatKnnFloatVectorQuery(String field, float[] target, int k, Query filter) {
    super(field, target, k, filter);
    LuceneVersionGuard.ensureCompatible();
  }

  /**
   * Returns the collector for searching one segment. It may hold fewer hits than asked for, but
   * never fewer than the query's {@code k}.
   *
   * @param context the segment
   * @param k the number of hits to collect
   * @param visitedLimit the visit budget
   */
  abstract KnnCollector newPerLeafCollector(LeafReaderContext context, int k, int visitedLimit)
      throws IOException;

  /**
   * Returns how many hits Lucene expects from searching a segment with a filter. Lucene 10.2 searches
   * a segment exactly when the filter accepts at most this many documents in it, and it does so too
   * when an approximate search returns fewer hits. On Lucene 10.2 this is {@code k}.
   *
   * @param context the segment
   */
  int perLeafTopK(LeafReaderContext context) {
    return k;
  }

  @Override
  protected TopDocs approximateSearch(
      LeafReaderContext context,
      Bits acceptDocs,
      int visitedLimit,
      KnnCollectorManager knnCollectorManager)
      throws IOException {
    KnnCollector results = newPerLeafCollector(context, k, visitedLimit);
    context.reader().searchNearestVectors(field, getTargetCopy(), results, acceptDocs);
    return results.topDocs();
  }
}

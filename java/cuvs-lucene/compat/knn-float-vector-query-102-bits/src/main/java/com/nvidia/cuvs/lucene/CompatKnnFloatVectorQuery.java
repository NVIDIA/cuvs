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
import org.apache.lucene.search.TopDocsCollector;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.util.Bits;

/**
 * Base class of the GPU query, {@code GPUKnnFloatVectorQuery}, that hides a signature change in
 * {@link KnnFloatVectorQuery}. Lucene 10.2 passes the documents a segment search may return to
 * {@code approximateSearch} as {@link Bits}; Lucene 10.3 changed that parameter to
 * {@code AcceptDocs}. Each variant of this class implements the {@code approximateSearch} of its
 * Lucene release with the collector from {@code newPerLeafCollector}, so the shared query only
 * provides that collector and {@link #maxHits}, the most hits a segment search can return. A
 * filtered search that cannot return as many hits as Lucene expects is skipped, since Lucene would
 * discard them; see {@link PerLeafK}.
 *
 * <p>Each variant also tells the shared query how many hits Lucene expects from a segment search,
 * {@link #perLeafTopK}: Lucene 10.2 expects {@code k}, Lucene 10.3 the segment's share of {@code
 * k} plus a margin, often more than {@code k} but less for a large {@code k} over many segments.
 */
abstract class CompatKnnFloatVectorQuery extends KnnFloatVectorQuery {

  CompatKnnFloatVectorQuery(String field, float[] target, int k, Query filter) {
    super(field, target, k, filter);
    LuceneVersionGuard.ensureCompatible();
  }

  /**
   * Returns the collector for searching one segment.
   *
   * @param leafK the number of hits to collect
   * @param visitedLimit the visit budget
   */
  abstract KnnCollector newPerLeafCollector(int leafK, int visitedLimit);

  /**
   * Returns the most hits a search of the segment returns, or {@link Integer#MAX_VALUE} if it has
   * no such limit.
   *
   * @param context the segment
   */
  abstract int maxHits(LeafReaderContext context);

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
    int leafK = PerLeafK.of(this, context, visitedLimit);
    if (leafK == PerLeafK.SKIP) {
      // The segment cannot return as many hits as Lucene needs, so Lucene would discard them and
      // search the segment exactly anyway: skip the approximate search and let it.
      return TopDocsCollector.EMPTY_TOPDOCS;
    }
    KnnCollector results = newPerLeafCollector(leafK, visitedLimit);
    context.reader().searchNearestVectors(field, getTargetCopy(), results, acceptDocs);
    return results.topDocs();
  }
}

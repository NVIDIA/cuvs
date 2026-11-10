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
import org.apache.lucene.search.TopDocsCollector;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.search.knn.KnnCollectorManager;

/**
 * Base class of the GPU query, {@code GPUKnnFloatVectorQuery}, that hides a signature change in
 * {@link KnnFloatVectorQuery}. Since Lucene 10.3, the documents a segment search may return come
 * to {@code approximateSearch} as {@link AcceptDocs}, where Lucene 10.2 passed {@code Bits}. Each
 * variant of this class implements the {@code approximateSearch} of its Lucene release with the
 * collector from {@code newPerLeafCollector}, so the shared query only provides that collector and
 * {@link #maxHits}, the most hits a segment search can return. A filtered search that cannot return
 * as many hits as Lucene expects is skipped, since Lucene would discard them; see {@link PerLeafK}.
 *
 * <p>Each variant also tells the shared query how many hits Lucene expects from a segment search,
 * {@link #perLeafTopK}: Lucene 10.2 expects {@code k}, Lucene 10.3 the segment's share of {@code
 * k} plus a margin, often more than {@code k} but less for a large {@code k} over many segments.
 *
 * <p>This variant also makes the collector manager non-optimistic, so that Lucene does not repeat
 * a segment's search; see {@link #getKnnCollectorManager}.
 */
abstract class CompatKnnFloatVectorQuery extends KnnFloatVectorQuery {

  CompatKnnFloatVectorQuery(String field, float[] target, int k, Query filter) {
    super(field, target, k, filter);
    LuceneVersionGuard.ensureCompatible();
  }

  // AbstractKnnVectorQuery.LAMBDA, which is private.
  private static final int LAMBDA = 16;

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
   * Returns how many hits Lucene expects from searching a segment with a filter. Since Lucene 10.3,
   * it searches a segment exactly when the filter accepts at most this many documents in it, and
   * it does so too when an approximate search returns fewer hits, throwing that search's results
   * away.
   *
   * <p>This is the segment's expected share of the top {@code k} hits plus a margin, computed like
   * Lucene's private {@code AbstractKnnVectorQuery.perLeafTopKCalculation}, so it is often more
   * than {@code k}.
   *
   * @param context the segment
   */
  int perLeafTopK(LeafReaderContext context) {
    float leafProportion =
        context.parent == null
            ? 1f
            : context.reader().maxDoc() / (float) context.parent.reader().maxDoc();
    return (int)
        Math.max(
            1, k * leafProportion + LAMBDA * Math.sqrt(k * leafProportion * (1 - leafProportion)));
  }

  /**
   * Returns a manager that is not optimistic. Since Lucene 10.3, {@code rewrite} searches again,
   * expecting more hits, every segment whose hits all beat the global top-k when the manager is
   * optimistic. {@link #approximateSearch} does not use the manager, so that second search would
   * only repeat the first one.
   */
  @Override
  protected KnnCollectorManager getKnnCollectorManager(int k, IndexSearcher searcher) {
    // A lambda does not override isOptimistic(), which defaults to false. approximateSearch never
    // calls it, so it builds no collector until then.
    return (visitedLimit, searchStrategy, context) ->
        new TopKnnCollector(k, visitedLimit, searchStrategy);
  }

  @Override
  protected TopDocs approximateSearch(
      LeafReaderContext context,
      AcceptDocs acceptDocs,
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

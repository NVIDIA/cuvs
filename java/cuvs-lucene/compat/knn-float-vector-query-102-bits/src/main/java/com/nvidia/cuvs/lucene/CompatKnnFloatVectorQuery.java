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
 */
abstract class CompatKnnFloatVectorQuery extends KnnFloatVectorQuery {

  CompatKnnFloatVectorQuery(String field, float[] target, int k, Query filter) {
    super(field, target, k, filter);
    LuceneVersionGuard.ensureCompatible();
  }

  /** Returns the collector for searching one segment, given its visit budget. */
  abstract KnnCollector newPerLeafCollector(int visitedLimit);

  @Override
  protected TopDocs approximateSearch(
      LeafReaderContext context,
      Bits acceptDocs,
      int visitedLimit,
      KnnCollectorManager knnCollectorManager)
      throws IOException {
    KnnCollector results = newPerLeafCollector(visitedLimit);
    context.reader().searchNearestVectors(field, getTargetCopy(), results, acceptDocs);
    return results.topDocs();
  }
}

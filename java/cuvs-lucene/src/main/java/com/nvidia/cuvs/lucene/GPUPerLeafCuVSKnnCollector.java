/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.knn.KnnSearchStrategy;

/**
 * KnnCollector for cuVS used for search on the GPU.
 *
 * @since 25.10
 */
class GPUPerLeafCuVSKnnCollector implements KnnCollector {

  public List<ScoreDoc> scoreDocs;
  public int topK = 0;
  public int iTopK = topK; // TODO getter, no setter
  public int searchWidth = 1; // TODO getter, no setter
  public int results = 0;

  /**
   * Initializes {@link GPUPerLeafCuVSKnnCollector}
   *
   * @param topK the topk value
   * @param iTopK the iTopK value
   * @param searchWidth the search width
   */
  public GPUPerLeafCuVSKnnCollector(int topK, int iTopK, int searchWidth) {
    super();
    this.topK = topK;
    this.iTopK = iTopK;
    this.searchWidth = searchWidth;
    scoreDocs = new ArrayList<ScoreDoc>();
  }

  @Override
  public boolean earlyTerminated() {
    // TODO: may need implementation
    return false;
  }

  @Override
  public void incVisitedCount(int count) {
    // TODO: may need implementation
  }

  @Override
  public long visitedCount() {
    // TODO: may need implementation
    return 0;
  }

  @Override
  public long visitLimit() {
    // TODO: may need implementation
    return 0;
  }

  @Override
  public int k() {
    return topK;
  }

  @Override
  @SuppressWarnings("cast")
  public boolean collect(int docId, float similarity) {
    scoreDocs.add(new ScoreDoc(docId, similarity));
    return true;
  }

  @Override
  public float minCompetitiveSimilarity() {
    // TODO: may need implementation
    return 0;
  }

  @Override
  public TopDocs topDocs() {
    return new TopDocs(
        new TotalHits(scoreDocs.size(), TotalHits.Relation.EQUAL_TO),
        scoreDocs.toArray(new ScoreDoc[scoreDocs.size()]));
  }

  @Override
  public KnnSearchStrategy getSearchStrategy() {
    return KnnSearchStrategy.Patience.DEFAULT;
  }
}

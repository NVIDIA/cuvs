/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.util.Bits;

/**
 * Extends upon KnnFloatVectorQuery for GPU-only search.
 *
 * @since 25.10
 */
public class GPUKnnFloatVectorQuery extends KnnFloatVectorQuery {

  private final int iTopK;
  private final int searchWidth;

  /**
   * Initializes {@link GPUKnnFloatVectorQuery}
   *
   * @param field the vector field name
   * @param target the vector target query
   * @param k the topK value
   * @param filter instance of the Query
   * @param iTopK the iTopK value
   * @param searchWidth the search width
   */
  public GPUKnnFloatVectorQuery(
      String field, float[] target, int k, Query filter, int iTopK, int searchWidth) {
    super(field, target, k, filter);
    this.iTopK = iTopK;
    this.searchWidth = searchWidth;
  }

  @Override
  protected TopDocs approximateSearch(
      LeafReaderContext context,
      Bits acceptDocs,
      int visitedLimit,
      KnnCollectorManager knnCollectorManager)
      throws IOException {

    KnnCollector results = new GPUPerLeafCuVSKnnCollector(k, iTopK, searchWidth);

    LeafReader reader = context.reader();
    reader.searchNearestVectors(field, this.getTargetCopy(), results, acceptDocs);
    return results.topDocs();
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import org.apache.lucene.search.TopKnnCollector;

/**
 * KnnCollector for cuVS used for search on the GPU.
 *
 * @since 25.10
 */
class GPUPerLeafCuVSKnnCollector extends TopKnnCollector {

  private int iTopK;
  private int searchWidth;

  /**
   * Initializes {@link GPUPerLeafCuVSKnnCollector}
   *
   * @param topK the topK value
   * @param iTopK the iTopK value
   * @param searchWidth the search width
   */
  public GPUPerLeafCuVSKnnCollector(int topK, int visitLimit, int iTopK, int searchWidth) {
    super(topK, visitLimit);
    this.iTopK = iTopK > topK ? iTopK : topK;
    this.searchWidth = searchWidth;
  }

  public int getiTopK() {
    return iTopK;
  }

  public int getSearchWidth() {
    return searchWidth;
  }
}

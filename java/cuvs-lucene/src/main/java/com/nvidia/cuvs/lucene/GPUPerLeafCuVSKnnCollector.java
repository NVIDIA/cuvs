/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CagraSearchParams;
import org.apache.lucene.search.TopKnnCollector;

/**
 * KnnCollector for cuVS used for search on the GPU.
 *
 * @since 25.10
 */
class GPUPerLeafCuVSKnnCollector extends TopKnnCollector {

  private int iTopK;
  private int searchWidth;
  private int threadBlockSize;
  private int maxIterations;
  private CagraSearchParams.SearchAlgo searchAlgo;

  /**
   * Initializes {@link GPUPerLeafCuVSKnnCollector}
   *
   * @param topK the topK value
   * @param iTopK the iTopK value
   * @param searchWidth the search width
   * @param threadBlockSize CAGRA thread_block_size (0 = auto; controls worker_queue_size)
   * @param maxIterations CAGRA max_iterations (0 = auto)
   * @param searchAlgo the CAGRA search algorithm
   */
  public GPUPerLeafCuVSKnnCollector(
      int topK,
      int visitLimit,
      int iTopK,
      int searchWidth,
      int threadBlockSize,
      int maxIterations,
      CagraSearchParams.SearchAlgo searchAlgo) {
    super(topK, visitLimit);
    this.iTopK = iTopK > topK ? iTopK : topK;
    this.searchWidth = searchWidth;
    this.threadBlockSize = threadBlockSize;
    this.maxIterations = maxIterations;
    this.searchAlgo = searchAlgo;
  }

  public int getiTopK() {
    return iTopK;
  }

  public int getSearchWidth() {
    return searchWidth;
  }

  public int getThreadBlockSize() {
    return threadBlockSize;
  }

  public int getMaxIterations() {
    return maxIterations;
  }

  public CagraSearchParams.SearchAlgo getSearchAlgo() {
    return searchAlgo;
  }
}

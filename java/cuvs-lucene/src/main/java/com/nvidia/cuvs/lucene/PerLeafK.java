/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import org.apache.lucene.index.LeafReaderContext;

/**
 * How many hits {@link CompatKnnFloatVectorQuery} collects from one segment. It is the same rule on
 * every Lucene release, so every variant of that class calls this instead of copying it.
 */
final class PerLeafK {

  /** Returned by {@link #of} when the segment search should be skipped. */
  static final int SKIP = -1;

  private PerLeafK() {}

  /**
   * Returns how many hits to collect from a segment, or {@link #SKIP} if Lucene would throw them
   * away: it keeps the hits of a filtered search only if there are {@link
   * CompatKnnFloatVectorQuery#perLeafTopK} of them, and searches the segment exactly otherwise. It
   * passes no visit limit when it applies no filter, and then keeps whatever the search returns.
   *
   * @param query the query
   * @param context the segment
   * @param visitedLimit the visit limit Lucene passed to {@code approximateSearch}
   */
  static int of(CompatKnnFloatVectorQuery query, LeafReaderContext context, int visitedLimit) {
    int k = query.getK();
    if (visitedLimit == Integer.MAX_VALUE) {
      return k;
    }
    int perLeafTopK = query.perLeafTopK(context);
    if (query.maxHits(context) < perLeafTopK) {
      return SKIP;
    }
    // At least k, since all of the global top k may come from this segment.
    return Math.max(k, perLeafTopK);
  }
}

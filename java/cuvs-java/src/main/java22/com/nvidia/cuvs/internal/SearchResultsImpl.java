/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal;

import com.nvidia.cuvs.SearchResults;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.function.LongToIntFunction;

class SearchResultsImpl implements SearchResults {

  private final List<Map<Integer, Float>> results;

  SearchResultsImpl(List<Map<Integer, Float>> results) {
    this.results = results;
  }

  /**
   * Factory method to create an on-heap SearchResults (backed by standard Java data types and containers) from
   * native/off-heap memory data structures.
   */
  static SearchResults create(
      SequenceLayout neighboursSequenceLayout,
      SequenceLayout distancesSequenceLayout,
      MemorySegment neighboursMemorySegment,
      MemorySegment distancesMemorySegment,
      int topK,
      LongToIntFunction mapping,
      long numberOfQueries) {
    List<Map<Integer, Float>> results = new LinkedList<>();
    var neighboursVarHandle =
        neighboursSequenceLayout.varHandle(MemoryLayout.PathElement.sequenceElement());
    var distancesVarHandle =
        distancesSequenceLayout.varHandle(MemoryLayout.PathElement.sequenceElement());

    // One map per query, so callers can rely on the result list holding exactly numberOfQueries
    // entries even when topK is 0 and no map has any content.
    for (long query = 0; query < numberOfQueries; query++) {
      Map<Integer, Float> resultMap = new LinkedHashMap<>();
      for (int j = 0; j < topK; j++) {
        long i = query * topK + j;
        long id = (long) neighboursVarHandle.get(neighboursMemorySegment, 0L, i);
        float dst = (float) distancesVarHandle.get(distancesMemorySegment, 0L, i);
        resultMap.put(mapping != null ? mapping.applyAsInt((int) id) : (int) id, dst);
      }
      results.add(resultMap);
    }

    return new SearchResultsImpl(results);
  }

  /**
   * Gets a list results as a map of neighbor IDs to distances.
   *
   * @return a list of results for each query as a map of neighbor IDs to distance
   */
  @Override
  public List<Map<Integer, Float>> getResults() {
    return results;
  }
}

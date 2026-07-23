/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.carrotsearch.randomizedtesting.RandomizedRunner;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@RunWith(RandomizedRunner.class)
public class MultiPartitionCagraSearchIT extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(MultiPartitionCagraSearchIT.class);

  private static final int NUM_PARTITIONS = 3;
  private static final int PART_ROWS = 40; // >> topK and intermediate_graph_degree -> no sentinels
  private static final int DIM = 16;
  private static final int NUM_QUERIES = 4;
  private static final int TOP_K = 10;
  private static final int N_ROWS = NUM_PARTITIONS * PART_ROWS;

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxAmd64());
    initializeRandom();
    log.trace("Random context initialized for test.");
  }

  /**
   * Searches N partitions (disjoint contiguous slices of one dataset) and verifies the merged
   * results match a brute-force top-k over the full dataset once (partitionIndex, ordinal) is
   * decoded back to a global index. Compares neighbor indices (metric-agnostic) rather than raw
   * distances.
   */
  @Test
  public void testMultiPartitionSearch() throws Throwable {
    float[][] dataset = generateData(random, N_ROWS, DIM);
    float[][] queries = generateData(random, NUM_QUERIES, DIM);
    int[] partStart = partitionStarts();

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      List<CagraIndex> indices = buildPartitions(dataset, partStart, resources);
      // Host-resident query vectors (uploaded to device internally); the filtered test below
      // exercises the device-resident query path.
      try (var queryVectors = CuVSMatrix.ofArray(queries)) {
        CagraQuery query =
            new CagraQuery.Builder(resources)
                .withTopK(TOP_K)
                .withSearchParams(new CagraSearchParams.Builder().build())
                .withQueryVectors(queryVectors)
                .build();

        MultiPartitionSearchResults results =
            MultiPartitionCagraSearch.search(resources, indices, query, TOP_K);

        // With partitions much larger than topK, every query fills topK -> no compaction.
        assertEquals(NUM_QUERIES * TOP_K, results.count());

        List<List<Integer>> actual = decodeToGlobalIds(results, partStart);
        List<List<Integer>> expected = generateExpectedResults(TOP_K, dataset, queries, null, log);
        assertTopResultsInExpected(actual, expected);
      } finally {
        closeAll(indices);
      }
    }
  }

  /**
   * Filtered multi-partition search via per-partition {@link FilterBitsetHandle}s (one bitset per
   * partition over its own rows); clearing the first {@code removeCount} global rows removes them.
   * Verified against a brute-force search with the matching prefilter.
   */
  @Test
  public void testMultiPartitionFilteredSearch() throws Throwable {
    final int removeCount = 12; // remove global rows [0, removeCount)

    float[][] dataset = generateData(random, N_ROWS, DIM);
    float[][] queries = generateData(random, NUM_QUERIES, DIM);
    int[] partStart = partitionStarts();

    // Brute-force ground truth with the same kept set (prefilter bit set == keep).
    BitSet keep = new BitSet(N_ROWS);
    keep.set(removeCount, N_ROWS);
    BitSet[] prefilters = new BitSet[NUM_QUERIES];
    Arrays.fill(prefilters, keep);

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      List<CagraIndex> indices = buildPartitions(dataset, partStart, resources);
      // One bitset per partition (one bit per local row; set == keep). Clearing the first
      // removeCount global rows removes them from their partition.
      final int longsPerPart = (PART_ROWS + 63) / 64;
      List<FilterBitsetHandle> filters = new ArrayList<>(NUM_PARTITIONS);
      for (int p = 0; p < NUM_PARTITIONS; p++) {
        long[] partLongs = new long[longsPerPart];
        for (int i = 0; i < PART_ROWS; i++) {
          if (p * PART_ROWS + i >= removeCount) { // keep this row
            partLongs[i / 64] |= 1L << (i % 64);
          }
        }
        filters.add(FilterBitsetHandle.create(partLongs));
      }
      // Device-resident query vectors here (the unfiltered test above covers host-resident).
      try (var hostQueries = CuVSMatrix.ofArray(queries);
          var queryVectors = hostQueries.toDevice(resources)) {
        CagraQuery query =
            new CagraQuery.Builder(resources)
                .withTopK(TOP_K)
                .withSearchParams(new CagraSearchParams.Builder().build())
                .withQueryVectors(queryVectors)
                .build();

        MultiPartitionSearchResults results =
            MultiPartitionCagraSearch.search(resources, indices, query, TOP_K, filters);

        assertEquals(NUM_QUERIES * TOP_K, results.count());

        // No filtered-out row may appear in the results.
        for (int i = 0; i < results.count(); i++) {
          int global = partStart[results.getPartitionIndex(i)] + results.getOrdinal(i);
          assertTrue("filtered-out row appeared in results: " + global, global >= removeCount);
        }

        List<List<Integer>> actual = decodeToGlobalIds(results, partStart);
        List<List<Integer>> expected =
            generateExpectedResults(TOP_K, dataset, queries, prefilters, log);
        assertTopResultsInExpected(actual, expected);
      } finally {
        for (FilterBitsetHandle f : filters) {
          f.close();
        }
        closeAll(indices);
      }
    }
  }

  // --- helpers ---

  private static int[] partitionStarts() {
    int[] starts = new int[NUM_PARTITIONS];
    for (int p = 0; p < NUM_PARTITIONS; p++) {
      starts[p] = p * PART_ROWS;
    }
    return starts;
  }

  private List<CagraIndex> buildPartitions(
      float[][] dataset, int[] partStart, CuVSResources resources) throws Throwable {
    CagraIndexParams indexParams =
        new CagraIndexParams.Builder()
            .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
            .withGraphDegree(16)
            .withIntermediateGraphDegree(32)
            .withMetric(CuvsDistanceType.L2Expanded)
            .build();

    List<CagraIndex> indices = new ArrayList<>();
    for (int p = 0; p < partStart.length; p++) {
      float[][] slice = Arrays.copyOfRange(dataset, partStart[p], partStart[p] + PART_ROWS);
      indices.add(
          CagraIndex.newBuilder(resources).withDataset(slice).withIndexParams(indexParams).build());
    }
    return indices;
  }

  /**
   * Decodes the flat results into per-query global-id lists in rank (distance) order. Relies on
   * count == NUM_QUERIES * TOP_K (asserted by the caller), so result {@code i} belongs to query
   * {@code i / TOP_K}.
   */
  private List<List<Integer>> decodeToGlobalIds(
      MultiPartitionSearchResults results, int[] partStart) {
    List<List<Integer>> perQuery = new ArrayList<>();
    for (int q = 0; q < NUM_QUERIES; q++) {
      perQuery.add(new ArrayList<>());
    }
    for (int i = 0; i < results.count(); i++) {
      int global = partStart[results.getPartitionIndex(i)] + results.getOrdinal(i);
      perQuery.get(i / TOP_K).add(global);
    }
    return perQuery;
  }

  /**
   * Recall-style check (mirrors CuVSTestCase.compareResults): the top results per query must appear
   * in the brute-force candidate set (generateExpectedResults returns ~2*topK+10 candidates).
   */
  private void assertTopResultsInExpected(
      List<List<Integer>> actual, List<List<Integer>> expected) {
    for (int q = 0; q < NUM_QUERIES; q++) {
      List<Integer> got = actual.get(q);
      List<Integer> exp = expected.get(q);
      int check = Math.min(5, got.size());
      for (int j = 0; j < check; j++) {
        assertTrue(
            "query " + q + ": neighbor " + got.get(j) + " not in expected set " + exp,
            exp.contains(got.get(j)));
      }
    }
  }

  private static void closeAll(List<CagraIndex> indices) throws Exception {
    for (CagraIndex idx : indices) {
      idx.close();
    }
  }
}

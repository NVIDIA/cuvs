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
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Builds CAGRA indexes over random datasets and compares the search results with an exact search
 * on the CPU.
 *
 * <p>CAGRA is an approximate index, and on uniformly random data in hundreds or thousands of
 * dimensions the nearest neighbors are barely closer than the rest, so a single query can
 * legitimately miss some or all of its exact neighbors. The test therefore checks what must hold
 * for every query (the right number of distinct, valid rows that pass the prefilter, returned in
 * order with their correct distances), and checks recall only as a mean over many queries. Each
 * case also searches for a copy of one of its rows, which the search should return first.
 *
 * <p>This test used to require the first 5 results of every query to be among the exact {@code
 * 2k + 10} nearest neighbors. That failed now and then for legitimate approximate results (#631),
 * and it also caught the Java bindings leaking filtered-out rows (#1048, #1468, fixed in #2375),
 * which the prefilter check here reports directly.
 */
@RunWith(RandomizedRunner.class)
public class CagraRandomizedIT extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(CagraRandomizedIT.class);

  // Lower bounds for the mean recall@k over the queries of each dataset memory kind, without and
  // with a prefilter. In CI (RTX PRO 6000, CUDA 13), about 300 runs of this test measured means of
  // 0.83 (standard deviation 0.022, lowest 0.76) without a prefilter and 0.94 (standard deviation
  // 0.011, lowest 0.90) with one. The bounds sit 8 and 12 standard deviations below the means, so
  // they only fail when results are wrong rather than approximate.
  private static final double MIN_MEAN_RECALL_UNFILTERED = 0.65;
  private static final double MIN_MEAN_RECALL_FILTERED = 0.8;
  // Minimum fraction of the searches for a copy of a row that return that row first. All of more
  // than 100,000 such searches did in CI.
  private static final double MIN_SELF_MATCH_RATE = 0.99;
  // Returned distances are computed in float on the GPU; allow for the difference in rounding
  // from the double-precision distances computed here (at most 2.4e-7 in CI).
  private static final double DISTANCE_RELATIVE_TOLERANCE = 1e-4;

  enum TestDatasetMemoryKind {
    HEAP,
    NATIVE,
    DEVICE
  }

  /** Recall accumulated per dataset memory kind, with and without a prefilter. */
  private final double[][] recallSum = new double[TestDatasetMemoryKind.values().length][2];

  private final int[][] queryCount = new int[TestDatasetMemoryKind.values().length][2];

  private int selfQueries;
  private int selfMatches;

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxSupportedArch());
    initializeRandom();
  }

  @Test
  public void testResultsTopKWithRandomValues() throws Throwable {
    for (int i = 0; i < 100; i++) {
      for (var datasetMemoryKind : TestDatasetMemoryKind.values()) {
        tmpResultsTopKWithRandomValues(datasetMemoryKind);
      }
    }

    assertTrue(
        String.format(
            "Only %d of %d searches for a row returned it first", selfMatches, selfQueries),
        selfMatches >= MIN_SELF_MATCH_RATE * selfQueries);
    for (var kind : TestDatasetMemoryKind.values()) {
      for (int filtered = 0; filtered < 2; filtered++) {
        int n = queryCount[kind.ordinal()][filtered];
        if (n == 0) {
          continue;
        }
        double meanRecall = recallSum[kind.ordinal()][filtered] / n;
        double minMeanRecall =
            filtered == 1 ? MIN_MEAN_RECALL_FILTERED : MIN_MEAN_RECALL_UNFILTERED;
        log.info(
            "{} {} prefilter: mean recall@k {} over {} queries",
            kind,
            filtered == 1 ? "with" : "without",
            meanRecall,
            n);
        assertTrue(
            String.format(
                "Mean recall@k %f over %d queries (%s, prefilter=%b) is below %f",
                meanRecall, n, kind, filtered == 1, minMeanRecall),
            meanRecall >= minMeanRecall);
      }
    }
  }

  /**
   * Gives the index the device-padded vectors CAGRA search needs. A device matrix whose rows are
   * already padded can only be viewed, not copied.
   */
  private static AutoCloseable attachPaddedDataset(CagraIndex index, CuVSMatrix deviceVectors)
      throws Throwable {
    if (CagraIndex.isPaddedDataset(deviceVectors)) {
      var view = index.makePaddedDatasetView(deviceVectors);
      index.updateDataset(view);
      return view;
    }
    var dataset = index.makePaddedDataset(deviceVectors);
    index.updateDataset(dataset);
    return dataset;
  }

  private void tmpResultsTopKWithRandomValues(TestDatasetMemoryKind datasetMemoryKind)
      throws Throwable {
    int DATASET_SIZE_LIMIT = 10_000;
    int DIMENSIONS_LIMIT = 2048;
    int NUM_QUERIES_LIMIT = 10;
    // The default CAGRA search parameters (itopk_size = 64) bound the number of results.
    int TOP_K_LIMIT = 64;

    // A CAGRA graph needs at least 2 rows.
    int datasetSize = random.nextInt(DATASET_SIZE_LIMIT) + 2;
    int dimensions = random.nextInt(DIMENSIONS_LIMIT) + 1;
    int numQueries = random.nextInt(NUM_QUERIES_LIMIT) + 1;
    int topK = Math.min(random.nextInt(TOP_K_LIMIT) + 1, datasetSize);
    boolean usePrefilter = random.nextBoolean();

    BitSet prefilter = null;
    if (usePrefilter) {
      prefilter = new BitSet(datasetSize);
      for (int j = 0; j < datasetSize; j++) {
        prefilter.set(j, random.nextBoolean());
      }
    }

    float[][] vectors = generateData(random, datasetSize, dimensions);
    float[][] queries = generateData(random, numQueries, dimensions);
    // One more query is a copy of a row that passes the prefilter, if any does.
    int selfRow = random.nextInt(datasetSize);
    if (prefilter != null && !prefilter.get(selfRow)) {
      int next = prefilter.nextSetBit(selfRow);
      selfRow = next >= 0 ? next : prefilter.previousSetBit(selfRow);
    }
    if (selfRow >= 0) {
      queries = Arrays.copyOf(queries, numQueries + 1);
      queries[numQueries] = vectors[selfRow].clone();
    }

    String caseDescription =
        String.format(
            "dataset %dx%d, %d queries, k=%d, prefilter=%b, %s",
            datasetSize, dimensions, numQueries, topK, usePrefilter, datasetMemoryKind);
    log.debug("Case: {}", caseDescription);

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      CagraIndexParams indexParams =
          new CagraIndexParams.Builder()
              .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
              .build();

      final CagraIndex index;
      if (datasetMemoryKind == TestDatasetMemoryKind.NATIVE) {
        var datasetBuilder =
            CuVSMatrix.hostBuilder(vectors.length, vectors[0].length, CuVSMatrix.DataType.FLOAT);
        for (float[] v : vectors) {
          datasetBuilder.addVector(v);
        }
        index =
            CagraIndex.newBuilder(resources)
                .fromDataset(datasetBuilder.build())
                .withIndexParams(indexParams)
                .build();
      } else if (datasetMemoryKind == TestDatasetMemoryKind.DEVICE) {
        var datasetBuilder =
            CuVSMatrix.deviceBuilder(
                resources, vectors.length, vectors[0].length, CuVSMatrix.DataType.FLOAT);
        for (float[] v : vectors) {
          datasetBuilder.addVector(v);
        }
        index =
            CagraIndex.newBuilder(resources)
                .fromDataset(datasetBuilder.build())
                .withIndexParams(indexParams)
                .build();
      } else {
        index =
            CagraIndex.newBuilder(resources)
                .fromDataset(vectors)
                .withIndexParams(indexParams)
                .build();
      }

      // CAGRA search needs the device-padded vectors, whatever memory the index was built from.
      try (var hostVectors = CuVSMatrix.ofArray(vectors);
          var deviceVectors = hostVectors.toDevice(resources);
          var indexDataset = attachPaddedDataset(index, deviceVectors);
          var queryVectors = CuVSMatrix.ofArray(queries)) {
        CagraQuery.Builder queryBuilder =
            new CagraQuery.Builder(resources)
                .withQueryVectors(queryVectors)
                .withTopK(topK)
                .withSearchParams(new CagraSearchParams.Builder().build());
        if (prefilter != null) {
          queryBuilder.withPrefilter(prefilter, datasetSize);
        }
        SearchResults results = index.search(queryBuilder.build());
        checkResults(
            caseDescription,
            datasetMemoryKind,
            results,
            vectors,
            queries,
            numQueries,
            prefilter,
            topK);
        if (selfRow >= 0) {
          var selfResult = results.getResults().get(numQueries);
          selfQueries++;
          if (!selfResult.isEmpty() && selfResult.keySet().iterator().next() == selfRow) {
            selfMatches++;
          }
        }
      } finally {
        index.close();
      }
    }
  }

  /** Squared L2 distances from the query to every row, +inf for the rows the filter removes. */
  private static double[] exactDistances(float[] query, float[][] vectors, BitSet prefilter) {
    double[] distances = new double[vectors.length];
    for (int j = 0; j < vectors.length; j++) {
      if (prefilter != null && !prefilter.get(j)) {
        distances[j] = Double.POSITIVE_INFINITY;
        continue;
      }
      double sum = 0;
      for (int d = 0; d < query.length; d++) {
        double diff = (double) query[d] - (double) vectors[j][d];
        sum += diff * diff;
      }
      distances[j] = sum;
    }
    return distances;
  }

  private void checkResults(
      String caseDescription,
      TestDatasetMemoryKind datasetMemoryKind,
      SearchResults results,
      float[][] vectors,
      float[][] queries,
      int numRandomQueries,
      BitSet prefilter,
      int topK) {
    List<Map<Integer, Float>> resultMaps = results.getResults();
    assertEquals(caseDescription, queries.length, resultMaps.size());
    int numPassing = prefilter == null ? vectors.length : prefilter.cardinality();
    int expectedCount = Math.min(topK, numPassing);
    int filtered = prefilter == null ? 0 : 1;

    for (int q = 0; q < queries.length; q++) {
      String where = caseDescription + ", query " + q;
      double[] exact = exactDistances(queries[q], vectors, prefilter);
      double[] sorted = exact.clone();
      Arrays.sort(sorted);

      // The results map holds one entry per distinct row, so the count also rules out duplicates.
      Map<Integer, Float> result = resultMaps.get(q);
      assertEquals(where + ": number of distinct results", expectedCount, result.size());

      float previous = Float.NEGATIVE_INFINITY;
      int found = 0;
      for (var entry : result.entrySet()) {
        int id = entry.getKey();
        float distance = entry.getValue();
        assertTrue(where + ": invalid row id " + id, id >= 0 && id < vectors.length);
        assertTrue(
            where + ": row " + id + " is excluded by the prefilter",
            prefilter == null || prefilter.get(id));
        assertTrue(where + ": results are not sorted by distance", distance >= previous);
        previous = distance;
        double tolerance = DISTANCE_RELATIVE_TOLERANCE * Math.max(exact[id], 1.0);
        assertTrue(
            where + ": distance " + distance + " of row " + id + " should be " + exact[id],
            Math.abs(distance - exact[id]) <= tolerance);
        // Rows exactly as close as the k-th nearest neighbor count as found.
        if (exact[id] <= sorted[expectedCount - 1]) {
          found++;
        }
      }
      // Recall is measured on the random queries only, not on the copy of a row.
      if (expectedCount > 0 && q < numRandomQueries) {
        recallSum[datasetMemoryKind.ordinal()][filtered] += (double) found / expectedCount;
        queryCount[datasetMemoryKind.ordinal()][filtered]++;
      }
    }
  }
}

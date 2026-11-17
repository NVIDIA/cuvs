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

@RunWith(RandomizedRunner.class)
public class CagraRandomizedIT extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(CagraRandomizedIT.class);

  // CAGRA is approximate, and on uniformly random data in high dimensions a single query can
  // legitimately miss some of its exact neighbors (#631), so recall is checked as a mean over every
  // query of the test. The means were about 0.83 without a prefilter and 0.94 with one in CI, and
  // somewhat lower on other GPUs; these bounds only fail when results are wrong, not approximate.
  private static final double MIN_MEAN_RECALL_UNFILTERED = 0.65;
  private static final double MIN_MEAN_RECALL_FILTERED = 0.7;
  // Allows for the GPU computing distances in float rather than double.
  private static final double DISTANCE_RELATIVE_TOLERANCE = 1e-4;

  // Indexed by whether the query used a prefilter.
  private final double[] recallSum = new double[2];
  private final int[] queryCount = new int[2];

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxSupportedArch());
    initializeRandom();
    log.trace("Random context initialized for test.");
  }

  enum TestDatasetMemoryKind {
    HEAP,
    NATIVE,
    DEVICE
  }

  @Test
  public void testResultsTopKWithRandomValues() throws Throwable {
    TestDatasetMemoryKind[] testDatasetMemoryKinds = {
      TestDatasetMemoryKind.HEAP, TestDatasetMemoryKind.NATIVE, TestDatasetMemoryKind.DEVICE
    };
    for (int i = 0; i < 30; i++) {
      for (var datasetMemoryKind : testDatasetMemoryKinds) {
        tmpResultsTopKWithRandomValues(datasetMemoryKind);
      }
    }

    for (int filtered = 0; filtered < 2; filtered++) {
      if (queryCount[filtered] == 0) {
        continue;
      }
      double meanRecall = recallSum[filtered] / queryCount[filtered];
      double minMeanRecall = filtered == 1 ? MIN_MEAN_RECALL_FILTERED : MIN_MEAN_RECALL_UNFILTERED;
      String description =
          String.format(
              "mean recall@k over %d queries (prefilter=%b)", queryCount[filtered], filtered == 1);
      log.info("{}: {}", description, meanRecall);
      assertTrue(description + " is " + meanRecall, meanRecall >= minMeanRecall);
    }
  }

  private void tmpResultsTopKWithRandomValues(TestDatasetMemoryKind datasetMemoryKind)
      throws Throwable {
    int DATASET_SIZE_LIMIT = 10_000;
    int DIMENSIONS_LIMIT = 2048;
    int NUM_QUERIES_LIMIT = 10;
    int TOP_K_LIMIT = 64; // nocommit This fails beyond 64

    int datasetSize =
        random.nextInt(DATASET_SIZE_LIMIT) + 2; // A CAGRA graph needs at least 2 rows.
    int dimensions = random.nextInt(DIMENSIONS_LIMIT) + 1;
    int numQueries = random.nextInt(NUM_QUERIES_LIMIT) + 1;
    int topK = Math.min(random.nextInt(TOP_K_LIMIT) + 1, datasetSize);
    boolean usePrefilter = random.nextBoolean();

    if (datasetSize < topK) datasetSize = topK;

    BitSet sharedPrefilter = null;
    if (usePrefilter) {
      sharedPrefilter = new BitSet(datasetSize);
      for (int j = 0; j < datasetSize; j++) {
        sharedPrefilter.set(j, random.nextBoolean());
      }
    }

    // Generate a random dataset
    float[][] vectors = generateData(random, datasetSize, dimensions);

    // Generate random query vectors
    float[][] queries = generateData(random, numQueries, dimensions);

    log.debug("Dataset size: {}x{}", datasetSize, dimensions);
    log.debug("Query size: {}x{}", numQueries, dimensions);
    log.debug("TopK: {}", topK);
    log.debug("Use memory dataset: " + datasetMemoryKind.name());

    // Debugging: Log dataset and queries
    if (log.isDebugEnabled()) {
      log.debug("Dataset:");
      for (float[] row : vectors) {
        log.debug(java.util.Arrays.toString(row));
      }
      log.debug("Queries:");
      for (float[] query : queries) {
        log.debug(java.util.Arrays.toString(query));
      }
    }
    // Sanity checks
    assert vectors.length > 0 : "Dataset is empty.";
    assert queries.length > 0 : "Queries are empty.";
    assert dimensions > 0 : "Invalid dimensions.";
    assert topK > 0 && topK <= datasetSize : "Invalid topK value.";

    // Create CuVS index and query
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
        assert datasetMemoryKind == TestDatasetMemoryKind.HEAP;
        index =
            CagraIndex.newBuilder(resources)
                .fromDataset(vectors)
                .withIndexParams(indexParams)
                .build();
      }
      log.trace("Index built successfully.");

      // CAGRA search needs the dataset padded on the device, whatever it was built from.
      try (var hostVectors = CuVSMatrix.ofArray(vectors);
          var deviceVectors = hostVectors.toDevice(resources);
          var indexDataset = attachPaddedDataset(index, deviceVectors);
          var queryVectors = CuVSMatrix.ofArray(queries)) {
        // Execute search and retrieve results
        CagraQuery.Builder queryBuilder =
            new CagraQuery.Builder(resources)
                .withQueryVectors(queryVectors)
                .withTopK(topK)
                .withSearchParams(new CagraSearchParams.Builder().build());

        if (sharedPrefilter != null) {
          queryBuilder.withPrefilter(sharedPrefilter, datasetSize);
        }

        CagraQuery query = queryBuilder.build();
        log.trace("Query built successfully. Executing search...");
        SearchResults results = index.search(query);

        checkResults(results, vectors, queries, sharedPrefilter, topK);
      } finally {
        index.close();
      }
    }
  }

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

  /**
   * Checks what must hold for every query: the right number of distinct, valid rows that pass the
   * prefilter, sorted by distance, with their exact distances. Recall is accumulated for the
   * check at the end of the test.
   */
  private void checkResults(
      SearchResults results, float[][] vectors, float[][] queries, BitSet prefilter, int topK) {
    List<Map<Integer, Float>> resultMaps = results.getResults();
    assertEquals(queries.length, resultMaps.size());
    int expectedCount =
        Math.min(topK, prefilter == null ? vectors.length : prefilter.cardinality());
    int filtered = prefilter == null ? 0 : 1;

    for (int q = 0; q < queries.length; q++) {
      double[] exact = new double[vectors.length];
      for (int j = 0; j < vectors.length; j++) {
        exact[j] = Double.POSITIVE_INFINITY;
        if (prefilter == null || prefilter.get(j)) {
          exact[j] = 0;
          for (int d = 0; d < queries[q].length; d++) {
            double diff = (double) queries[q][d] - vectors[j][d];
            exact[j] += diff * diff;
          }
        }
      }
      double[] sorted = exact.clone();
      Arrays.sort(sorted);

      // A map holds one entry per row, so the count also rules out duplicates.
      Map<Integer, Float> result = resultMaps.get(q);
      assertEquals("query " + q + ": " + result, expectedCount, result.size());
      float previous = Float.NEGATIVE_INFINITY;
      int found = 0;
      for (var entry : result.entrySet()) {
        int id = entry.getKey();
        float distance = entry.getValue();
        String where = "query " + q + ", row " + id;
        assertTrue(where + ": invalid row id", id >= 0 && id < vectors.length);
        assertTrue(where + ": excluded by the prefilter", prefilter == null || prefilter.get(id));
        assertTrue(where + ": results are not sorted by distance", distance >= previous);
        assertEquals(
            where + ": distance",
            exact[id],
            distance,
            DISTANCE_RELATIVE_TOLERANCE * Math.max(exact[id], 1.0));
        previous = distance;
        // Rows as close as the k-th nearest neighbor count as found.
        if (exact[id] <= sorted[expectedCount - 1]) {
          found++;
        }
      }
      if (expectedCount > 0) {
        recallSum[filtered] += (double) found / expectedCount;
        queryCount[filtered]++;
      }
    }
  }
}

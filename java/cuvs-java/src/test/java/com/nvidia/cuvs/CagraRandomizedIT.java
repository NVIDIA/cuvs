/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;

import com.carrotsearch.randomizedtesting.RandomizedRunner;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import java.util.BitSet;
import java.util.List;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@RunWith(RandomizedRunner.class)
public class CagraRandomizedIT extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(CagraRandomizedIT.class);

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxAmd64());
    initializeRandom();
    log.trace("Random context initialized for test.");
  }

  enum TestDatasetMemoryKind {
    HEAP,
    NATIVE,
    DEVICE
  }

  @Ignore // https://github.com/nvidia/cuvs/issues/1468
  @Test
  public void testResultsTopKWithRandomValues() throws Throwable {
    TestDatasetMemoryKind[] testDatasetMemoryKinds = {
      TestDatasetMemoryKind.HEAP, TestDatasetMemoryKind.NATIVE, TestDatasetMemoryKind.DEVICE
    };
    for (int i = 0; i < 100; i++) {
      for (var datasetMemoryKind : testDatasetMemoryKinds) {
        tmpResultsTopKWithRandomValues(datasetMemoryKind);
      }
    }
  }

  private void tmpResultsTopKWithRandomValues(TestDatasetMemoryKind datasetMemoryKind)
      throws Throwable {
    int DATASET_SIZE_LIMIT = 10_000;
    int DIMENSIONS_LIMIT = 2048;
    int NUM_QUERIES_LIMIT = 10;
    int TOP_K_LIMIT = 64; // nocommit This fails beyond 64

    int datasetSize =
        random.nextInt(DATASET_SIZE_LIMIT)
            + 2; // datasetSize of 1 fails/crashed due to a bug, hence adding 2 here.
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

    BitSet[] prefilters = null;
    if (sharedPrefilter != null) {
      prefilters = new BitSet[numQueries];
      for (int i = 0; i < numQueries; i++) {
        prefilters[i] = sharedPrefilter;
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

    // Generate expected results using brute force
    List<List<Integer>> expected = generateExpectedResults(topK, vectors, queries, prefilters, log);

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
                .withDataset(datasetBuilder.build())
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
                .withDataset(datasetBuilder.build())
                .withIndexParams(indexParams)
                .build();
      } else {
        assert datasetMemoryKind == TestDatasetMemoryKind.HEAP;
        index =
            CagraIndex.newBuilder(resources)
                .withDataset(vectors)
                .withIndexParams(indexParams)
                .build();
      }
      log.trace("Index built successfully.");

      try (var queryVectors = CuVSMatrix.ofArray(queries)) {
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

        compareResults(results, expected, topK, datasetSize, numQueries);
      } finally {
        index.close();
      }
    }
  }
}

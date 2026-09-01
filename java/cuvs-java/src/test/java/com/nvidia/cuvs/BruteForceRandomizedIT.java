/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;

import com.carrotsearch.randomizedtesting.RandomizedRunner;
import com.nvidia.cuvs.spi.CuVSProvider;
import java.lang.invoke.MethodHandles;
import java.util.BitSet;
import java.util.List;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@RunWith(RandomizedRunner.class)
public class BruteForceRandomizedIT extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

  @Before
  public void setup() {
    assumeTrue(isLinuxSupportedArch());
    initializeRandom();
    log.trace("Random context initialized for test.");
    CuVSProvider.provider().enableRMMPooledMemory(10, 60);
  }

  @After
  public void cleanup() {
    CuVSProvider.provider().resetRMMPooledMemory();
  }

  @Test
  public void testResultsTopKWithRandomValues() throws Throwable {
    boolean useNativeMemoryDatasets[] = {true, false};
    for (int i = 0; i < 10; i++) {
      for (boolean use : useNativeMemoryDatasets) {
        tmpResultsTopKWithRandomValues(use);
      }
    }
  }

  private void tmpResultsTopKWithRandomValues(boolean useNativeMemoryDataset) throws Throwable {
    int DATASET_SIZE_LIMIT = 10_000;
    int DIMENSIONS_LIMIT = 2048;
    int NUM_QUERIES_LIMIT = 10;
    int TOP_K_LIMIT = 64; // nocommit This fails beyond 64

    int datasetSize = random.nextInt(DATASET_SIZE_LIMIT) + 1;
    int dimensions = random.nextInt(DIMENSIONS_LIMIT) + 1;
    int numQueries = random.nextInt(NUM_QUERIES_LIMIT) + 1;
    int topK = Math.min(random.nextInt(TOP_K_LIMIT) + 1, datasetSize);
    boolean usePrefilter = random.nextBoolean();
    if (datasetSize < topK) datasetSize = topK;

    BitSet[] prefilters = null;
    if (usePrefilter) {
      prefilters = new BitSet[numQueries];
      for (int i = 0; i < numQueries; i++) {
        BitSet randomFilter = new BitSet(datasetSize);
        for (int j = 0; j < datasetSize; j++) {
          randomFilter.set(j, random.nextBoolean());
        }
        prefilters[i] = randomFilter;
      }
    }

    // Generate a random dataset
    float[][] vectors = generateData(random, datasetSize, dimensions);

    // Generate random query vectors
    float[][] queries = generateData(random, numQueries, dimensions);

    log.debug("Dataset size: {}x{}", datasetSize, dimensions);
    log.debug("Query size: {}x{}", numQueries, dimensions);
    log.debug("TopK: {}", topK);
    log.debug("Use native memory dataset? " + useNativeMemoryDataset);

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

      BruteForceQuery query =
          new BruteForceQuery.Builder(resources)
              .withTopK(topK)
              .withQueryVectors(queries)
              .withPrefilters(prefilters, vectors.length)
              .build();

      BruteForceIndexParams indexParams =
          new BruteForceIndexParams.Builder().withNumWriterThreads(32).build();

      BruteForceIndex index;
      if (useNativeMemoryDataset) {
        var datasetBuilder =
            CuVSMatrix.hostBuilder(vectors.length, vectors[0].length, CuVSMatrix.DataType.FLOAT);
        for (float[] v : vectors) {
          datasetBuilder.addVector(v);
        }
        index =
            BruteForceIndex.newBuilder(resources)
                .withDataset(datasetBuilder.build())
                .withIndexParams(indexParams)
                .build();
      } else {
        index =
            BruteForceIndex.newBuilder(resources)
                .withDataset(vectors)
                .withIndexParams(indexParams)
                .build();
      }

      log.trace("Index built successfully. Executing search...");
      SearchResults results = index.search(query);

      compareResults(results, expected, topK, datasetSize, numQueries);
    }
  }
}

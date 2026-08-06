/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;

import com.carrotsearch.randomizedtesting.RandomizedRunner;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import com.nvidia.cuvs.HnswIndexParams.HnswHierarchy;
import java.lang.invoke.MethodHandles;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.UUID;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@RunWith(RandomizedRunner.class)
public class HnswRandomizedIT extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxAmd64());
    initializeRandom();
    log.trace("Random context initialized for test.");
  }

  @Test
  public void testResultsTopKWithRandomValues() throws Throwable {
    boolean[] useNativeMemoryDatasets = {true, false};
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

    if (datasetSize < topK) datasetSize = topK;

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
    List<List<Integer>> expected = generateExpectedResults(topK, vectors, queries, null, log);

    // Create CuVS index and query
    try (CuVSResources resources = CheckedCuVSResources.create()) {

      // Configure index parameters
      CagraIndexParams indexParams =
          new CagraIndexParams.Builder()
              .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
              .withGraphDegree(64)
              .withIntermediateGraphDegree(128)
              .withNumWriterThreads(32)
              .withMetric(CuvsDistanceType.L2Expanded)
              .build();

      // Create the index with the dataset
      final CagraIndex index;
      if (useNativeMemoryDataset) {
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
      } else {
        index =
            CagraIndex.newBuilder(resources)
                .withDataset(vectors)
                .withIndexParams(indexParams)
                .build();
      }

      // Saving the HNSW index on to the disk.
      String hnswIndexFileName = UUID.randomUUID() + ".hnsw";
      var hnswIndexPath = Path.of(hnswIndexFileName);

      try {
        try (var outputStream = Files.newOutputStream(hnswIndexPath)) {
          index.serializeToHNSW(outputStream); // fails here
        }

        // Use NONE hierarchy since serializeToHNSW creates a base-layer-only index
        HnswIndexParams hnswIndexParams =
            new HnswIndexParams.Builder()
                .withVectorDimension(dimensions)
                .withHierarchy(HnswHierarchy.NONE)
                .build();

        try (var inputStreamHNSW = Files.newInputStream(hnswIndexPath)) {
          HnswIndex hnswIndex =
              HnswIndex.newBuilder(resources)
                  .from(inputStreamHNSW)
                  .withIndexParams(hnswIndexParams)
                  .build();

          HnswSearchParams hnswSearchParams =
              new HnswSearchParams.Builder().withNumThreads(32).build();

          HnswQuery hnswQuery =
              new HnswQuery.Builder(resources)
                  .withQueryVectors(queries)
                  .withSearchParams(hnswSearchParams)
                  .withTopK(topK)
                  .build();

          log.trace("Index built successfully. Executing search...");
          SearchResults results = hnswIndex.search(hnswQuery);
          compareResults(results, expected, topK, datasetSize, numQueries);

          hnswIndex.close();
        }
      } finally {
        index.close();
        Files.deleteIfExists(hnswIndexPath);
      }
    }
  }
}

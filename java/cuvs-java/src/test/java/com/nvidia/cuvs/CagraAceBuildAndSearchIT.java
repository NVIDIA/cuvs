/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;
import static org.junit.Assert.*;

import com.carrotsearch.randomizedtesting.RandomizedRunner;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import com.nvidia.cuvs.HnswIndexParams.HnswHierarchy;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Integration tests for CAGRA index using ACE (Augmented Core Extraction) build algorithm.
 * ACE enables building indexes for datasets too large to fit in GPU memory by partitioning
 * the dataset and building sub-indexes.
 *
 * @since 25.12
 */
@RunWith(RandomizedRunner.class)
public class CagraAceBuildAndSearchIT extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(CagraAceBuildAndSearchIT.class);

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxAmd64());
    initializeRandom();
    log.trace("Random context initialized for test.");
  }

  private static List<Map<Integer, Float>> getExpectedResults() {
    return Arrays.asList(
        Map.of(3, 0.038782578f, 2, 0.3590463f, 0, 0.83774555f),
        Map.of(0, 0.12472608f, 2, 0.21700792f, 1, 0.31918612f),
        Map.of(3, 0.047766715f, 2, 0.20332818f, 0, 0.48305473f),
        Map.of(1, 0.15224178f, 0, 0.59063464f, 3, 0.5986642f));
  }

  private static float[][] createSampleQueries() {
    return new float[][] {
      {0.48216683f, 0.0428398f},
      {0.5084142f, 0.6545497f},
      {0.51260436f, 0.2643005f},
      {0.05198065f, 0.5789965f}
    };
  }

  private static float[][] createSampleData() {
    return new float[][] {
      {0.74021935f, 0.9209938f},
      {0.03902049f, 0.9689629f},
      {0.92514056f, 0.4463501f},
      {0.6673192f, 0.10993068f}
    };
  }

  /**
   * Test ACE build with in-memory mode (use_disk=false).
   * This tests the basic ACE functionality with small datasets that fit in memory.
   */
  @Test
  public void testAceInMemoryBuild() throws Throwable {
    float[][] dataset = createSampleData();
    float[][] queries = createSampleQueries();
    List<Map<Integer, Float>> expectedResults = getExpectedResults();

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      // Configure ACE parameters for in-memory mode
      CuVSAceParams aceParams =
          new CuVSAceParams.Builder()
              .withNpartitions(2)
              .withEfConstruction(120)
              .withUseDisk(false)
              .build();

      // Configure index parameters with ACE build algorithm
      CagraIndexParams indexParams =
          new CagraIndexParams.Builder()
              .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.ACE)
              .withGraphDegree(64)
              .withIntermediateGraphDegree(128)
              .withNumWriterThreads(2)
              .withMetric(CuvsDistanceType.L2Expanded)
              .withCuVSAceParams(aceParams)
              .build();

      // Build the index with ACE
      try (CagraIndex index =
              CagraIndex.newBuilder(resources)
                  .withDataset(dataset)
                  .withIndexParams(indexParams)
                  .build();
          var hostVectors = CuVSMatrix.ofArray(dataset);
          var deviceVectors = hostVectors.toDevice(resources);
          var indexDataset = index.makePaddedDataset(deviceVectors)) {
        index.updateDataset(indexDataset);

        // Verify index was built
        assertNotNull("Index should not be null", index);
        log.debug("ACE index built successfully in memory");

        // Perform search
        CagraSearchParams searchParams = new CagraSearchParams.Builder().build();

        try (var queryVectors = CuVSMatrix.ofArray(queries)) {
          CagraQuery cuvsQuery =
              new CagraQuery.Builder(resources)
                  .withTopK(3)
                  .withSearchParams(searchParams)
                  .withQueryVectors(queryVectors)
                  .build();

          SearchResults results = index.search(cuvsQuery);
          log.debug("Search results: " + results.getResults().toString());

          // Verify search results
          checkResults(expectedResults, results.getResults());
        }
      }
    }
  }

  /**
   * Test ACE build with disk-based mode (use_disk=true).
   * This tests ACE's ability to handle large datasets that don't fit in GPU memory.
   */
  @Test
  public void testAceDiskBasedBuild() throws Throwable {
    float[][] dataset = createSampleData();
    float[][] queries = createSampleQueries();
    List<Map<Integer, Float>> expectedResults = getExpectedResults();

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      // Configure ACE parameters for disk-based mode
      Path buildDir = Path.of("/tmp/java_ace_test");
      CuVSAceParams aceParams =
          new CuVSAceParams.Builder()
              .withNpartitions(2)
              .withEfConstruction(120)
              .withUseDisk(true)
              .withBuildDir(buildDir.toString())
              .build();

      // Configure index parameters with ACE build algorithm
      CagraIndexParams indexParams =
          new CagraIndexParams.Builder()
              .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.ACE)
              .withGraphDegree(64)
              .withIntermediateGraphDegree(128)
              .withNumWriterThreads(32)
              .withMetric(CuvsDistanceType.L2Expanded)
              .withCuVSAceParams(aceParams)
              .build();

      // Build the index with ACE in disk mode
      try (CagraIndex index =
          CagraIndex.newBuilder(resources)
              .withDataset(dataset)
              .withIndexParams(indexParams)
              .build()) {

        // Verify index was built
        assertNotNull("Index should not be null", index);
        log.debug("ACE index built successfully with disk mode");

        // Verify ACE created the expected output files in the build directory
        assertTrue(
            "CAGRA graph file should exist", Files.exists(buildDir.resolve("cagra_graph.npy")));
        assertTrue(
            "Reordered dataset file should exist",
            Files.exists(buildDir.resolve("reordered_dataset.npy")));
        assertTrue(
            "Dataset mapping file should exist",
            Files.exists(buildDir.resolve("dataset_mapping.npy")));

        log.debug("ACE disk output files verified");

        // Convert CAGRA index to HNSW using fromCagra
        // This automatically handles disk-based indices
        HnswIndexParams hnswIndexParams =
            new HnswIndexParams.Builder()
                .withVectorDimension(2)
                .withHierarchy(HnswHierarchy.GPU)
                .build();

        try (var hnswIndexSerialized = HnswIndex.fromCagra(hnswIndexParams, index)) {
          var hnswIndexSerializedPath = buildDir.resolve("hnsw_index.bin");
          assertTrue("HNSW index should exist", Files.exists(hnswIndexSerializedPath));
          log.debug("HNSW index created from disk-based ACE CAGRA index");

          // Load the serialized index from disk
          try (var inputStreamHNSW = Files.newInputStream(hnswIndexSerializedPath)) {
            var hnswIndex =
                HnswIndex.newBuilder(resources)
                    .from(inputStreamHNSW)
                    .withIndexParams(hnswIndexParams)
                    .build();

            HnswSearchParams hnswSearchParams = new HnswSearchParams.Builder().build();
            HnswQuery hnswQuery =
                new HnswQuery.Builder(resources)
                    .withTopK(3)
                    .withSearchParams(hnswSearchParams)
                    .withQueryVectors(queries)
                    .build();

            SearchResults results = hnswIndex.search(hnswQuery);
            log.debug("HNSW search results: " + results.getResults().toString());

            checkResults(expectedResults, results.getResults());
            log.debug("HNSW search verification passed");

            hnswIndex.close();
          }
        }

        // Clean up the default build directory
        deleteRecursively(buildDir);
      }
    }
  }

  /**
   * Helper method to recursively delete a directory and its contents.
   */
  private void deleteRecursively(Path path) {
    try {
      if (Files.isDirectory(path)) {
        Files.list(path).forEach(this::deleteRecursively);
      }
      Files.deleteIfExists(path);
    } catch (Exception e) {
      log.warn("Failed to delete {}: {}", path, e.getMessage());
    }
  }
}

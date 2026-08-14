/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;
import static org.junit.Assert.*;

import com.carrotsearch.randomizedtesting.RandomizedRunner;
import com.nvidia.cuvs.HnswIndexParams.CuvsDistanceType;
import com.nvidia.cuvs.HnswIndexParams.CuvsHnswHierarchy;
import java.io.InputStream;
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
 * Integration tests for HNSW index using ACE (Augmented Core Extraction) build algorithm.
 * ACE enables building HNSW indexes for datasets too large to fit in GPU memory.
 *
 * Tests follow the same approach as C++ and C tests:
 * - Build HNSW index using ACE via HnswIndex.build()
 * - For disk mode: serialize -> deserialize -> search
 * - For in-memory mode: search directly
 *
 * @since 25.12
 */
@RunWith(RandomizedRunner.class)
public class HnswAceBuildAndSearchIT extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(HnswAceBuildAndSearchIT.class);

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
   * Test HNSW ACE build with in-memory mode (use_disk=false).
   */
  @Test
  public void testHnswAceInMemoryBuild() throws Throwable {
    float[][] dataset = createSampleData();
    float[][] queries = createSampleQueries();
    List<Map<Integer, Float>> expectedResults = getExpectedResults();

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      Path buildDir = Files.createTempDirectory("hnsw_ace_test");

      try {
        // Configure ACE parameters for in-memory mode
        HnswAceParams aceParams =
            new HnswAceParams.Builder()
                .withNpartitions(2)
                .withBuildDir(buildDir.toString())
                .withUseDisk(false)
                .build();

        // Configure HNSW index parameters with ACE
        HnswIndexParams hnswParams =
            new HnswIndexParams.Builder()
                .withHierarchy(CuvsHnswHierarchy.GPU)
                .withM(16)
                .withEfConstruction(100)
                .withMetric(CuvsDistanceType.L2Expanded)
                .withVectorDimension(2)
                .withAceParams(aceParams)
                .build();

        // Build the HNSW index using ACE
        try (var datasetMatrix = CuVSMatrix.ofArray(dataset)) {
          HnswIndex hnswIndex = HnswIndex.build(resources, hnswParams, datasetMatrix);
          assertNotNull("HNSW index should not be null", hnswIndex);
          log.debug("HNSW ACE index built successfully in memory");

          // Search the index directly
          HnswSearchParams searchParams = new HnswSearchParams.Builder().withEF(100).build();
          HnswQuery hnswQuery =
              new HnswQuery.Builder(resources)
                  .withTopK(3)
                  .withSearchParams(searchParams)
                  .withQueryVectors(queries)
                  .build();

          SearchResults results = hnswIndex.search(hnswQuery);
          log.debug("HNSW ACE search results: " + results.getResults().toString());

          // Verify search results
          checkResults(expectedResults, results.getResults());
          log.debug("HNSW ACE in-memory search verification passed");

          hnswIndex.close();
        }
      } finally {
        deleteRecursively(buildDir);
      }
    }
  }

  /**
   * Test HNSW ACE build with disk-based mode (use_disk=true).
   * This follows the same approach as C++ and C tests:
   * build -> serialize -> deserialize -> search
   */
  @Test
  public void testHnswAceDiskBasedBuild() throws Throwable {
    float[][] dataset = createSampleData();
    float[][] queries = createSampleQueries();
    List<Map<Integer, Float>> expectedResults = getExpectedResults();

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      Path buildDir = Files.createTempDirectory("hnsw_ace_disk_test");

      try {
        // Configure ACE parameters for disk-based mode
        HnswAceParams aceParams =
            new HnswAceParams.Builder()
                .withNpartitions(2)
                .withBuildDir(buildDir.toString())
                .withUseDisk(true)
                .build();

        // Configure HNSW index parameters with ACE
        HnswIndexParams hnswParams =
            new HnswIndexParams.Builder()
                .withHierarchy(CuvsHnswHierarchy.GPU)
                .withM(16)
                .withEfConstruction(100)
                .withMetric(CuvsDistanceType.L2Expanded)
                .withVectorDimension(2)
                .withAceParams(aceParams)
                .build();

        // Build the HNSW index using ACE with disk mode
        try (var datasetMatrix = CuVSMatrix.ofArray(dataset)) {
          HnswIndex hnswIndex = HnswIndex.build(resources, hnswParams, datasetMatrix);
          assertNotNull("HNSW index should not be null", hnswIndex);
          log.debug("HNSW ACE index built with disk mode");

          // For disk mode, the hnsw_index.bin file is created during build
          Path hnswIndexPath = buildDir.resolve("hnsw_index.bin");

          // Verify the index file was created by the disk-based build
          assertTrue("HNSW index file should exist", Files.exists(hnswIndexPath));
          log.debug("HNSW index serialized to: " + hnswIndexPath);

          // Deserialize from disk for searching
          try (InputStream inputStream = Files.newInputStream(hnswIndexPath)) {
            HnswIndex deserializedIndex =
                HnswIndex.newBuilder(resources)
                    .from(inputStream)
                    .withIndexParams(hnswParams)
                    .build();

            assertNotNull("Deserialized index should not be null", deserializedIndex);
            log.debug("HNSW index deserialized from disk");

            // Search the deserialized index
            HnswSearchParams searchParams = new HnswSearchParams.Builder().withEF(100).build();
            HnswQuery hnswQuery =
                new HnswQuery.Builder(resources)
                    .withTopK(3)
                    .withSearchParams(searchParams)
                    .withQueryVectors(queries)
                    .build();

            SearchResults results = deserializedIndex.search(hnswQuery);
            log.debug("HNSW ACE disk search results: " + results.getResults().toString());

            // Verify search results
            checkResults(expectedResults, results.getResults());
            log.debug("HNSW ACE disk-based search verification passed");

            deserializedIndex.close();
          }

          hnswIndex.close();
        }
      } finally {
        deleteRecursively(buildDir);
      }
    }
  }

  /**
   * Test HNSW ACE build with different hierarchy options.
   */
  @Test
  public void testHnswAceWithDifferentHierarchy() throws Throwable {
    float[][] dataset = createSampleData();
    float[][] queries = createSampleQueries();
    List<Map<Integer, Float>> expectedResults = getExpectedResults();

    for (CuvsHnswHierarchy hierarchy :
        Arrays.asList(CuvsHnswHierarchy.NONE, CuvsHnswHierarchy.GPU)) {
      try (CuVSResources resources = CheckedCuVSResources.create()) {
        Path buildDir = Files.createTempDirectory("hnsw_ace_hierarchy_test");

        try {
          HnswAceParams aceParams =
              new HnswAceParams.Builder()
                  .withNpartitions(2)
                  .withBuildDir(buildDir.toString())
                  .withUseDisk(false)
                  .build();

          HnswIndexParams hnswParams =
              new HnswIndexParams.Builder()
                  .withHierarchy(hierarchy)
                  .withM(16)
                  .withEfConstruction(100)
                  .withMetric(CuvsDistanceType.L2Expanded)
                  .withVectorDimension(2)
                  .withAceParams(aceParams)
                  .build();

          try (var datasetMatrix = CuVSMatrix.ofArray(dataset)) {
            HnswIndex hnswIndex = HnswIndex.build(resources, hnswParams, datasetMatrix);
            assertNotNull("HNSW index should not be null for hierarchy: " + hierarchy, hnswIndex);

            HnswSearchParams searchParams = new HnswSearchParams.Builder().withEF(100).build();
            HnswQuery hnswQuery =
                new HnswQuery.Builder(resources)
                    .withTopK(3)
                    .withSearchParams(searchParams)
                    .withQueryVectors(queries)
                    .build();

            SearchResults results = hnswIndex.search(hnswQuery);
            checkResults(expectedResults, results.getResults());
            log.debug("HNSW ACE with hierarchy {} verification passed", hierarchy);

            hnswIndex.close();
          }
        } finally {
          deleteRecursively(buildDir);
        }
      }
    }
  }

  /**
   * Test the full disk-based ACE workflow explicitly.
   * build -> serialize -> deserialize -> search
   */
  @Test
  public void testHnswAceDiskSerializeDeserialize() throws Throwable {
    float[][] dataset = createSampleData();
    float[][] queries = createSampleQueries();
    List<Map<Integer, Float>> expectedResults = getExpectedResults();

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      Path buildDir = Files.createTempDirectory("hnsw_ace_serialize_test");

      try {
        // Create ACE params with disk mode enabled
        HnswAceParams aceParams =
            new HnswAceParams.Builder()
                .withNpartitions(2)
                .withBuildDir(buildDir.toString())
                .withUseDisk(true)
                .build();

        // Create HNSW index params with ACE
        HnswIndexParams hnswParams =
            new HnswIndexParams.Builder()
                .withHierarchy(CuvsHnswHierarchy.GPU)
                .withM(16)
                .withEfConstruction(100)
                .withMetric(CuvsDistanceType.L2Expanded)
                .withVectorDimension(2)
                .withAceParams(aceParams)
                .build();

        // Build the index using ACE
        try (var datasetMatrix = CuVSMatrix.ofArray(dataset)) {
          HnswIndex hnswIndex = HnswIndex.build(resources, hnswParams, datasetMatrix);
          assertNotNull("HNSW index should not be null", hnswIndex);

          // The disk-based build should create hnsw_index.bin
          Path hnswFile = buildDir.resolve("hnsw_index.bin");
          assertTrue("HNSW index file should exist after disk build", Files.exists(hnswFile));

          // Deserialize from disk
          try (InputStream inputStream = Files.newInputStream(hnswFile)) {
            HnswIndex loadedIndex =
                HnswIndex.newBuilder(resources)
                    .from(inputStream)
                    .withIndexParams(hnswParams)
                    .build();

            // Search the loaded index
            HnswSearchParams searchParams = new HnswSearchParams.Builder().withEF(200).build();
            HnswQuery hnswQuery =
                new HnswQuery.Builder(resources)
                    .withTopK(3)
                    .withSearchParams(searchParams)
                    .withQueryVectors(queries)
                    .build();

            SearchResults results = loadedIndex.search(hnswQuery);
            log.debug("Serialize/Deserialize test results: " + results.getResults().toString());

            // Verify results
            checkResults(expectedResults, results.getResults());
            log.debug("HNSW ACE serialize/deserialize test passed");

            loadedIndex.close();
          }

          hnswIndex.close();
        }
      } finally {
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

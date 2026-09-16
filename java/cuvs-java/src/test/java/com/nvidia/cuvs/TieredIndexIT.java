/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import com.carrotsearch.randomizedtesting.RandomizedRunner;
import com.nvidia.cuvs.spi.CuVSProvider;
import java.lang.invoke.MethodHandles;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@RunWith(RandomizedRunner.class)
public class TieredIndexIT extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxSupportedArch());
    initializeRandom();
    CuVSProvider.provider().enableRMMPooledMemory(10, 60);
    log.debug("Random context initialized for test");
  }

  @After
  public void cleanup() {
    CuVSProvider.provider().resetRMMPooledMemory();
  }

  @Test
  public void testBasicOperations() throws Throwable {
    float[][] initialDataset = {
      {0.0f, 0.0f},
      {1.0f, 1.0f},
      {2.0f, 2.0f}
    };

    float[][] queries = {
      {0.1f, 0.1f},
      {1.9f, 1.9f}
    };

    float[][] extensionVectors = {
      {3.0f, 3.0f},
      {4.0f, 4.0f}
    };

    List<Map<Integer, Float>> expectedInitialResults =
        Arrays.asList(Map.of(0, 0.02f, 1, 1.62f, 2, 7.22f), Map.of(2, 0.02f, 1, 1.62f, 0, 7.22f));

    List<Map<Integer, Float>> expectedExtendedResults =
        Arrays.asList(Map.of(0, 0.02f, 1, 1.62f, 2, 7.22f), Map.of(2, 0.02f, 3, 2.42f, 1, 1.62f));

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      CagraIndexParams cagraParams =
          new CagraIndexParams.Builder().withGraphDegree(4).withIntermediateGraphDegree(8).build();

      TieredIndexParams indexParams =
          new TieredIndexParams.Builder()
              .minAnnRows(2)
              .createAnnIndexOnExtend(true)
              .withCagraParams(cagraParams)
              .build();

      log.debug("Building initial TieredIndex with {} vectors", initialDataset.length);
      TieredIndex index =
          TieredIndex.newBuilder(resources)
              .withDataset(initialDataset)
              .withIndexParams(indexParams)
              .build();
      log.debug("Initial TieredIndex built successfully");

      CagraSearchParams searchParams =
          new CagraSearchParams.Builder().withMaxIterations(20).build();

      TieredIndexQuery query =
          TieredIndexQuery.newBuilder(resources)
              .withTopK(3)
              .withQueryVectors(queries)
              .withSearchParams(searchParams)
              .build();

      log.debug("Searching initial index with {} queries", queries.length);
      SearchResults initialResults = index.search(query);
      log.debug("Initial search completed, validating results");
      assertEquals(expectedInitialResults, roundResults(initialResults.getResults()));

      log.debug("Extending index with {} additional vectors", extensionVectors.length);
      index.extend().withDataset(extensionVectors).execute();
      log.debug("Index extension completed");

      log.debug("Searching extended index");
      SearchResults extendedResults = index.search(query);
      log.debug("Extended search completed, validating results");
      assertEquals(expectedExtendedResults, roundResults(extendedResults.getResults()));
    }
  }

  @Test(expected = IllegalArgumentException.class)
  public void testErrorHandling() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create()) {
      CagraIndexParams cagraParams =
          new CagraIndexParams.Builder().withGraphDegree(4).withIntermediateGraphDegree(8).build();

      TieredIndexParams indexParams =
          new TieredIndexParams.Builder().minAnnRows(2).withCagraParams(cagraParams).build();

      log.debug("Testing error handling with null dataset");
      TieredIndex.newBuilder(resources)
          .withIndexParams(indexParams)
          .withDataset((float[][]) null)
          .build();
    }
  }

  @Test
  public void testDifferentKValues() throws Throwable {
    float[][] dataset = {
      {0.0f, 0.0f},
      {1.0f, 1.0f},
      {2.0f, 2.0f},
      {3.0f, 3.0f},
      {4.0f, 4.0f}
    };

    float[][] queries = {{0.1f, 0.1f}};

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      CagraIndexParams cagraParams =
          new CagraIndexParams.Builder().withGraphDegree(4).withIntermediateGraphDegree(8).build();

      TieredIndexParams indexParams =
          new TieredIndexParams.Builder().minAnnRows(2).withCagraParams(cagraParams).build();

      log.debug("Building TieredIndex for K-value testing with {} vectors", dataset.length);
      TieredIndex index =
          TieredIndex.newBuilder(resources)
              .withDataset(dataset)
              .withIndexParams(indexParams)
              .build();
      log.debug("TieredIndex built for K-value testing");

      TieredIndexQuery query1 =
          TieredIndexQuery.newBuilder(resources)
              .withTopK(1)
              .withQueryVectors(queries)
              .withSearchParams(new CagraSearchParams.Builder().withMaxIterations(20).build())
              .build();

      log.debug("Searching with K=1");
      SearchResults results1 = index.search(query1);
      Map<Integer, Float> firstResult = results1.getResults().get(0);
      log.debug("K=1 search completed, found {} results", firstResult.size());

      assertEquals(1, firstResult.size());
      assertTrue("Should contain index 0 (closest vector)", firstResult.containsKey(0));
      assertEquals("Distance to closest vector should be ~0.02", 0.02f, firstResult.get(0), 0.01f);

      TieredIndexQuery query3 =
          TieredIndexQuery.newBuilder(resources)
              .withTopK(3)
              .withQueryVectors(queries)
              .withSearchParams(new CagraSearchParams.Builder().withMaxIterations(20).build())
              .build();

      log.debug("Searching with K=3");
      SearchResults results3 = index.search(query3);
      Map<Integer, Float> thirdResult = results3.getResults().get(0);
      log.debug("K=3 search completed, found {} results", thirdResult.size());

      assertEquals(3, thirdResult.size());
      assertTrue("Should contain index 0", thirdResult.containsKey(0));
      assertTrue("Should contain index 1", thirdResult.containsKey(1));
      assertTrue("Should contain index 2", thirdResult.containsKey(2));

      float dist0 = thirdResult.get(0);
      float dist1 = thirdResult.get(1);
      float dist2 = thirdResult.get(2);

      assertTrue("Distance to index 0 should be smallest", dist0 <= dist1 && dist0 <= dist2);
    }
  }

  @Test
  public void testPrefilter() throws Throwable {
    float[][] dataset = {{0.0f, 0.0f}, {1.0f, 1.0f}, {2.0f, 2.0f}, {3.0f, 3.0f}};
    float[][] queryVectors = {{0.1f, 0.1f}};

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      CagraIndexParams cagraParams =
          new CagraIndexParams.Builder().withGraphDegree(4).withIntermediateGraphDegree(8).build();

      TieredIndexParams indexParams =
          new TieredIndexParams.Builder().minAnnRows(2).withCagraParams(cagraParams).build();

      log.debug("Building TieredIndex for prefilter testing with {} vectors", dataset.length);
      TieredIndex index =
          TieredIndex.newBuilder(resources)
              .withDataset(dataset)
              .withIndexParams(indexParams)
              .build();
      log.debug("TieredIndex built for prefilter testing");

      CagraSearchParams searchParams = new CagraSearchParams.Builder().build();

      BitSet prefilter = new BitSet(4);
      prefilter.set(1, true);
      prefilter.set(2, true);
      log.debug("Created prefilter allowing indices 1 and 2, excluding 0 and 3");

      TieredIndexQuery queryWithFilter =
          TieredIndexQuery.newBuilder(resources)
              .withTopK(3)
              .withQueryVectors(queryVectors)
              .withSearchParams(searchParams)
              .withPrefilter(prefilter, 4)
              .build();

      log.debug("Searching with prefilter applied");
      SearchResults resultsWithFilter = index.search(queryWithFilter);
      Map<Integer, Float> result = resultsWithFilter.getResults().get(0);
      log.debug("Prefilter search completed, validating filtered results");

      assertFalse("Index 0 should be filtered out", result.containsKey(0));
      assertFalse("Index 3 should be filtered out", result.containsKey(3));
      assertTrue("Index 1 or 2 should be present", result.containsKey(1) || result.containsKey(2));
    }
  }

  private List<Map<Integer, Float>> roundResults(List<Map<Integer, Float>> results) {
    return results.stream()
        .map(
            queryResult ->
                queryResult.entrySet().stream()
                    .collect(
                        Collectors.toMap(
                            Map.Entry::getKey,
                            entry -> Math.round(entry.getValue() * 100.0f) / 100.0f)))
        .collect(Collectors.toList());
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static com.carrotsearch.randomizedtesting.RandomizedTest.*;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.carrotsearch.randomizedtesting.RandomizedContext;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakLingering;
import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Base class for the cuVS integration tests.
 *
 * <p>The lingering is not cosmetic. A {@link java.util.concurrent.ThreadPoolExecutor} reaches
 * TERMINATED as soon as its worker count drops to zero, which happens in {@code getTask()} before
 * the workers have finished {@code processWorkerExit()}, so {@code awaitTermination()} returns
 * while the worker threads are still alive. {@code ThreadLeakLingering} defaults to no wait at
 * all, which leaves the leak check sampling straight into that window and reporting threads that
 * are in the middle of dying. Waiting a few seconds for them costs nothing when there is no leak,
 * and a real one still fails the test.
 */
@ThreadLeakLingering(linger = 5000)
public abstract class CuVSTestCase {
  protected Random random;
  private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

  protected void initializeRandom() {
    random = RandomizedContext.current().getRandom();
    log.debug("Test seed: " + RandomizedContext.current().getRunnerSeedAsString());
  }

  protected float[][] generateData(Random random, int rows, int cols) {
    float[][] data = new float[rows][cols];
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        data[i][j] = random.nextFloat() * 100;
      }
    }
    return data;
  }

  protected List<List<Integer>> generateExpectedResults(
      int topK, float[][] dataset, float[][] queries, BitSet[] prefilters, Logger log) {
    List<List<Integer>> neighborsResult = new ArrayList<>();
    int dimensions = dataset[0].length;

    for (int q = 0; q < queries.length; q++) {
      float[] query = queries[q];
      Map<Integer, Double> distances = new TreeMap<>();
      for (int j = 0; j < dataset.length; j++) {
        double distance = 0;
        if (prefilters != null && !prefilters[q].get(j)) {
          distance = Double.POSITIVE_INFINITY;
        } else {
          for (int k = 0; k < dimensions; k++) {
            distance += (query[k] - dataset[j][k]) * (query[k] - dataset[j][k]);
          }
        }
        distances.put(j, Math.sqrt(distance));
      }

      // Sort by distance and select the topK nearest neighbors
      List<Integer> neighbors =
          distances.entrySet().stream()
              .sorted(Map.Entry.comparingByValue())
              .map(Map.Entry::getKey)
              .toList();
      neighborsResult.add(neighbors.subList(0, Math.min(topK * 2 + 10, dataset.length)));
    }

    log.trace("Expected results generated successfully.");
    return neighborsResult;
  }

  protected void compareResults(
      SearchResults results,
      List<List<Integer>> expected,
      int topK,
      int datasetSize,
      int numQueries) {

    for (int i = 0; i < numQueries; i++) {
      log.debug("Results returned for query " + i + ": " + results.getResults().get(i).keySet());
      log.debug(
          "Expected results for query "
              + i
              + ": "
              + expected.get(i).subList(0, Math.min(topK, datasetSize)));
    }

    // actual vs. expected results
    for (int i = 0; i < results.getResults().size(); i++) {
      Map<Integer, Float> result = results.getResults().get(i);

      // Sort result by values (distances) and extract keys
      List<Integer> sortedResultKeys =
          result.entrySet().stream()
              .sorted(Map.Entry.comparingByValue())
              .map(Map.Entry::getKey) // Extract sorted keys
              .toList();

      // just make sure that the first 5 results are in the expected list (which
      // consists of 2*topK results)
      for (int j = 0; j < Math.min(5, sortedResultKeys.size()); j++) {
        assertTrue(
            "Not found in expected list: " + sortedResultKeys.get(j),
            expected.get(i).contains(sortedResultKeys.get(j)));
      }
    }
  }

  protected static void checkResults(
      List<Map<Integer, Float>> expected, List<Map<Integer, Float>> actual) {
    List<Map<Integer, Float>> sortedExpected = new ArrayList<Map<Integer, Float>>();
    List<Map<Integer, Float>> sortedActual = new ArrayList<Map<Integer, Float>>();
    for (Map<Integer, Float> map : expected) {
      sortedExpected.add(
          new TreeMap<>(map) {
            @Override
            public boolean equals(Object o) {
              if (!(o instanceof Map<?, ?>)) {
                return false;
              }
              @SuppressWarnings("unchecked")
              var map = (Map<Integer, Float>) o;
              if (this.size() != map.size()) return false;
              for (Integer key : map.keySet()) {
                try {
                  if (Math.abs(map.get(key) - ((float) get(key))) >= 0.0001f) {
                    return false;
                  }
                } catch (Exception ex) {
                  return false;
                }
              }
              return true;
            }
          });
    }
    for (Map<Integer, Float> map : actual) {
      sortedActual.add(new TreeMap<>(map));
    }
    assertEquals(sortedExpected, sortedActual);
  }

  protected static boolean isLinuxAmd64() {
    String name = System.getProperty("os.name");
    return (name.startsWith("Linux")) && System.getProperty("os.arch").equals("amd64");
  }

  protected static int[][] createIntMatrix() {
    int rows = randomIntBetween(1, 32);
    int cols = randomIntBetween(1, 100);

    return createIntMatrix(rows, cols);
  }

  protected static int[][] createIntMatrix(int rows, int cols) {
    int[][] result = new int[rows][cols];

    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        result[r][c] = randomInt();
      }
    }
    return result;
  }

  protected static byte[][] createByteMatrix() {
    int rows = randomIntBetween(1, 32);
    int cols = randomIntBetween(1, 100);

    return createByteMatrix(rows, cols);
  }

  protected static byte[][] createByteMatrix(int rows, int cols) {
    byte[][] result = new byte[rows][cols];

    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        result[r][c] = randomByte();
      }
    }
    return result;
  }

  protected static float[][] createFloatMatrix() {
    int rows = randomIntBetween(1, 32);
    int cols = randomIntBetween(1, 100);

    return createFloatMatrix(rows, cols);
  }

  protected static float[][] createFloatMatrix(int rows, int cols) {
    float[][] result = new float[rows][cols];

    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        result[r][c] = randomFloat();
      }
    }
    return result;
  }
}

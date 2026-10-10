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

  // [DO NOT MERGE] Diagnostics for flaky randomized tests: per query, report where each of the
  // first 5 returned ids ranks in the exact (squared L2) ordering, and the tie-aware recall@k.
  protected void logMismatchDiagnostics(
      String tag,
      SearchResults results,
      float[][] dataset,
      float[][] queries,
      BitSet[] prefilters,
      int topK) {
    int n = dataset.length;
    int dim = dataset[0].length;
    for (int q = 0; q < results.getResults().size(); q++) {
      double[] d = new double[n];
      int nPass = 0;
      for (int j = 0; j < n; j++) {
        if (prefilters != null && !prefilters[q].get(j)) {
          d[j] = Double.POSITIVE_INFINITY;
          continue;
        }
        nPass++;
        double s = 0;
        for (int k = 0; k < dim; k++) {
          double t = (double) queries[q][k] - (double) dataset[j][k];
          s += t * t;
        }
        d[j] = s;
      }
      double[] sorted = d.clone();
      java.util.Arrays.sort(sorted);
      double kth = sorted[Math.min(topK, n) - 1];
      double win = sorted[Math.min(topK * 2 + 10, n) - 1];
      Map<Integer, Float> r = results.getResults().get(q);
      List<Map.Entry<Integer, Float>> sortedRes =
          r.entrySet().stream().sorted(Map.Entry.comparingByValue()).toList();
      int good = 0;
      for (var e : sortedRes) {
        int id = e.getKey();
        if (id >= 0 && id < n && d[id] <= kth) good++;
      }
      StringBuilder sb = new StringBuilder();
      sb.append(
          String.format(
              "DIAG %s q=%d n=%d dim=%d k=%d nPass=%d returned=%d recall@k=%.3f kth=%.4f"
                  + " win(2k+10)=%.4f",
              tag, q, n, dim, topK, nPass, r.size(), (double) good / topK, kth, win));
      for (int j = 0; j < Math.min(5, sortedRes.size()); j++) {
        int id = sortedRes.get(j).getKey();
        float got = sortedRes.get(j).getValue();
        if (id < 0 || id >= n) {
          sb.append(String.format(" | #%d id=%d INVALID got=%.4f", j, id, got));
          continue;
        }
        int rank = 0;
        int ties = 0;
        for (int t = 0; t < n; t++) {
          if (d[t] < d[id]) rank++;
          else if (d[t] == d[id] && t != id) ties++;
        }
        sb.append(
            String.format(
                " | #%d id=%d rank=%d ties=%d exact=%.4f got=%.4f pass=%b",
                j, id, rank, ties, d[id], got, prefilters == null || prefilters[q].get(id)));
      }
      log.info(sb.toString());
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

  protected static boolean isLinuxSupportedArch() {
    String name = System.getProperty("os.name");
    String arch = System.getProperty("os.arch");
    return (name.startsWith("Linux")) && (arch.equals("amd64") || arch.equals("aarch64"));
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

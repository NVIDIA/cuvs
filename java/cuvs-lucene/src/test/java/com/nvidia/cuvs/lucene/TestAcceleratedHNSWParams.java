/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.AcceleratedHNSWParams.DEFAULT_BEAM_WIDTH;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWParams.DEFAULT_CAGRA_GRAPH_BUILD_ALGO;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWParams.DEFAULT_CUVS_DISTANCE_TYPE;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWParams.DEFAULT_GRAPH_DEGREE;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWParams.DEFAULT_HNSW_LAYERS;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWParams.DEFAULT_INT_GRAPH_DEGREE;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWParams.DEFAULT_MAX_CONN;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWParams.DEFAULT_NN_DESCENT_NUM_ITERATIONS;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWParams.DEFAULT_NUM_MERGE_WORKERS;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWParams.DEFAULT_STRATEGY;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWParams.DEFAULT_WRITER_THREADS;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWParams.MAX_BEAM_WIDTH;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWParams.MAX_GRAPH_DEG;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWParams.MAX_HNSW_LAYERS;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWParams.MAX_INT_GRAPH_DEG;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWParams.MAX_MAX_CONN;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWParams.MAX_NN_DESCENT_NUM_ITERATIONS;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWParams.MAX_NUM_MERGE_WORKERS;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWParams.MAX_WRITER_THREADS;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWParams.MIN_BEAM_WIDTH;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWParams.MIN_GRAPH_DEG;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWParams.MIN_HNSW_LAYERS;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWParams.MIN_INT_GRAPH_DEG;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWParams.MIN_MAX_CONN;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWParams.MIN_NN_DESCENT_NUM_ITERATIONS;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWParams.MIN_NUM_MERGE_WORKERS;
import static com.nvidia.cuvs.lucene.AcceleratedHNSWParams.MIN_WRITER_THREADS;
import static java.lang.Integer.MAX_VALUE;
import static java.lang.Integer.MIN_VALUE;

import java.util.Random;
import java.util.logging.Logger;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.BeforeClass;
import org.junit.Test;

@SuppressSysoutChecks(bugUrl = "")
public class TestAcceleratedHNSWParams extends LuceneTestCase {

  @SuppressWarnings("unused")
  private static final Logger log = Logger.getLogger(TestAcceleratedHNSWParams.class.getName());

  private static Random random;

  @Test
  public void testAcceleratedHNSWParamsDefaultValues() {
    AcceleratedHNSWParams params = new AcceleratedHNSWParams.Builder().build();
    assertEquals(DEFAULT_BEAM_WIDTH, params.getBeamWidth());
    assertEquals(DEFAULT_GRAPH_DEGREE, params.getGraphdegree());
    assertEquals(DEFAULT_HNSW_LAYERS, params.getHnswLayers());
    assertEquals(DEFAULT_INT_GRAPH_DEGREE, params.getIntermediateGraphDegree());
    assertEquals(DEFAULT_MAX_CONN, params.getMaxConn());
    assertEquals(DEFAULT_WRITER_THREADS, params.getWriterThreads());
    assertEquals(DEFAULT_NUM_MERGE_WORKERS, params.getNumMergeWorkers());
    assertEquals(DEFAULT_CAGRA_GRAPH_BUILD_ALGO, params.getCagraGraphBuildAlgo());
    assertEquals(DEFAULT_STRATEGY, params.getStrategy());
    assertEquals(DEFAULT_CUVS_DISTANCE_TYPE, params.getCuvsDistanceType());
    assertEquals(DEFAULT_NN_DESCENT_NUM_ITERATIONS, params.getNNDescentNumIterations());
  }

  @Test
  public void testAcceleratedHNSWParamsInvalidBeamWidth() {
    for (int v :
        new int[] {
          random.nextInt(MIN_VALUE, MIN_BEAM_WIDTH), random.nextInt(MAX_BEAM_WIDTH + 1, MAX_VALUE)
        }) {
      assertThrows(
          IllegalArgumentException.class,
          () -> new AcceleratedHNSWParams.Builder().withBeamWidth(v).build());
    }
  }

  @Test
  public void testAcceleratedHNSWParamsInvalidGraphDegree() {
    for (int v :
        new int[] {
          random.nextInt(MIN_VALUE, MIN_GRAPH_DEG), random.nextInt(MAX_GRAPH_DEG + 1, MAX_VALUE)
        }) {
      assertThrows(
          IllegalArgumentException.class,
          () -> new AcceleratedHNSWParams.Builder().withGraphDegree(v).build());
    }
  }

  @Test
  public void testAcceleratedHNSWParamsInvalidHNSWLayers() {
    for (int v :
        new int[] {
          random.nextInt(MIN_VALUE, MIN_HNSW_LAYERS), random.nextInt(MAX_HNSW_LAYERS + 1, MAX_VALUE)
        }) {
      assertThrows(
          IllegalArgumentException.class,
          () -> new AcceleratedHNSWParams.Builder().withHNSWLayer(v).build());
    }
  }

  @Test
  public void testAcceleratedHNSWParamsInvalidIntGraphDegree() {
    for (int v :
        new int[] {
          random.nextInt(MIN_VALUE, MIN_INT_GRAPH_DEG),
          random.nextInt(MAX_INT_GRAPH_DEG + 1, MAX_VALUE)
        }) {
      assertThrows(
          IllegalArgumentException.class,
          () -> new AcceleratedHNSWParams.Builder().withIntermediateGraphDegree(v).build());
    }
  }

  @Test
  public void testAcceleratedHNSWParamsInvalidMaxConn() {
    for (int v :
        new int[] {
          random.nextInt(MIN_VALUE, MIN_MAX_CONN), random.nextInt(MAX_MAX_CONN + 1, MAX_VALUE)
        }) {
      assertThrows(
          IllegalArgumentException.class,
          () -> new AcceleratedHNSWParams.Builder().withMaxConn(v).build());
    }
  }

  @Test
  public void testAcceleratedHNSWParamsInvalidWriterThreads() {
    for (int v :
        new int[] {
          random.nextInt(MIN_VALUE, MIN_WRITER_THREADS),
          random.nextInt(MAX_WRITER_THREADS + 1, Integer.MAX_VALUE)
        }) {
      assertThrows(
          IllegalArgumentException.class,
          () -> new AcceleratedHNSWParams.Builder().withWriterThreads(v).build());
    }
  }

  @Test
  public void testAcceleratedHNSWParamsInvalidNumMergeWorkers() {
    for (int v :
        new int[] {
          random.nextInt(MIN_VALUE, MIN_NUM_MERGE_WORKERS),
          random.nextInt(MAX_NUM_MERGE_WORKERS + 1, MAX_VALUE)
        }) {
      assertThrows(
          IllegalArgumentException.class,
          () -> new AcceleratedHNSWParams.Builder().withNumMergeWorkers(v).build());
    }
  }

  @Test
  public void testAcceleratedHNSWParamsInvalidCagraGraphBuildAlgo() {
    assertThrows(
        IllegalArgumentException.class,
        () -> new AcceleratedHNSWParams.Builder().withCagraGraphBuildAlgo(null).build());
  }

  @Test
  public void testAcceleratedHNSWParamsInvalidStrategy() {
    assertThrows(
        IllegalArgumentException.class,
        () -> new AcceleratedHNSWParams.Builder().withStrategy(null).build());
  }

  @Test
  public void testAcceleratedHNSWParamsInvalidCuvsDistanceType() {
    assertThrows(
        IllegalArgumentException.class,
        () -> new AcceleratedHNSWParams.Builder().withCuvsDistanceType(null).build());
  }

  @Test
  public void testAcceleratedHNSWParamsInvalidNNDescentNumIterations() {
    for (int v :
        new int[] {
          random.nextInt(MIN_VALUE, (int) MIN_NN_DESCENT_NUM_ITERATIONS),
          random.nextInt((int) (MAX_NN_DESCENT_NUM_ITERATIONS + 1), MAX_VALUE)
        }) {
      assertThrows(
          IllegalArgumentException.class,
          () -> new AcceleratedHNSWParams.Builder().withNNDescentNumIterations(v).build());
    }
  }

  @BeforeClass
  public static void beforeClass() {
    random = random();
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.GPUSearchParams.DEFAULT_CAGRA_GRAPH_BUILD_ALGO;
import static com.nvidia.cuvs.lucene.GPUSearchParams.DEFAULT_CUVS_DISTANCE_TYPE;
import static com.nvidia.cuvs.lucene.GPUSearchParams.DEFAULT_GRAPH_DEGREE;
import static com.nvidia.cuvs.lucene.GPUSearchParams.DEFAULT_INDEX_TYPE;
import static com.nvidia.cuvs.lucene.GPUSearchParams.DEFAULT_INT_GRAPH_DEGREE;
import static com.nvidia.cuvs.lucene.GPUSearchParams.DEFAULT_NN_DESCENT_NUM_ITERATIONS;
import static com.nvidia.cuvs.lucene.GPUSearchParams.DEFAULT_STRATEGY;
import static com.nvidia.cuvs.lucene.GPUSearchParams.DEFAULT_WRITER_THREADS;
import static com.nvidia.cuvs.lucene.GPUSearchParams.MAX_GRAPH_DEG;
import static com.nvidia.cuvs.lucene.GPUSearchParams.MAX_INT_GRAPH_DEG;
import static com.nvidia.cuvs.lucene.GPUSearchParams.MAX_NN_DESCENT_NUM_ITERATIONS;
import static com.nvidia.cuvs.lucene.GPUSearchParams.MAX_WRITER_THREADS;
import static com.nvidia.cuvs.lucene.GPUSearchParams.MIN_GRAPH_DEG;
import static com.nvidia.cuvs.lucene.GPUSearchParams.MIN_INT_GRAPH_DEG;
import static com.nvidia.cuvs.lucene.GPUSearchParams.MIN_NN_DESCENT_NUM_ITERATIONS;
import static com.nvidia.cuvs.lucene.GPUSearchParams.MIN_WRITER_THREADS;
import static java.lang.Integer.MAX_VALUE;
import static java.lang.Integer.MIN_VALUE;

import java.util.Random;
import java.util.logging.Logger;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.BeforeClass;
import org.junit.Test;

@SuppressSysoutChecks(bugUrl = "")
public class TestGPUSearchParams extends LuceneTestCase {

  @SuppressWarnings("unused")
  private static final Logger log = Logger.getLogger(TestGPUSearchParams.class.getName());

  private static Random random;

  @Test
  public void testGPUSearchParamsDefaultValues() {
    GPUSearchParams params = new GPUSearchParams.Builder().build();
    assertEquals(DEFAULT_GRAPH_DEGREE, params.getGraphdegree());
    assertEquals(DEFAULT_INT_GRAPH_DEGREE, params.getIntermediateGraphDegree());
    assertEquals(DEFAULT_WRITER_THREADS, params.getWriterThreads());
    assertEquals(DEFAULT_CAGRA_GRAPH_BUILD_ALGO, params.getCagraGraphBuildAlgo());
    assertEquals(DEFAULT_INDEX_TYPE, params.getIndexType());
    assertEquals(DEFAULT_STRATEGY, params.getStrategy());
    assertEquals(DEFAULT_CUVS_DISTANCE_TYPE, params.getCuvsDistanceType());
    assertEquals(DEFAULT_NN_DESCENT_NUM_ITERATIONS, params.getnNDescentNumIterations());
  }

  @Test
  public void testGPUSearchParamsInvalidGraphDegree() {
    for (int v :
        new int[] {
          random.nextInt(MIN_VALUE, MIN_GRAPH_DEG), random.nextInt(MAX_GRAPH_DEG + 1, MAX_VALUE)
        }) {
      assertThrows(
          IllegalArgumentException.class,
          () -> new GPUSearchParams.Builder().withGraphDegree(v).build());
    }
  }

  @Test
  public void testGPUSearchParamsInvalidIntermediateGraphDegree() {
    for (int v :
        new int[] {
          random.nextInt(MIN_VALUE, MIN_INT_GRAPH_DEG),
          random.nextInt(MAX_INT_GRAPH_DEG + 1, MAX_VALUE)
        }) {
      assertThrows(
          IllegalArgumentException.class,
          () -> new GPUSearchParams.Builder().withIntermediateGraphDegree(v).build());
    }
  }

  @Test
  public void testGPUSearchParamsInvalidWriterThreads() {
    for (int v :
        new int[] {
          random.nextInt(MIN_VALUE, MIN_WRITER_THREADS),
          random.nextInt(MAX_WRITER_THREADS + 1, MAX_VALUE)
        }) {
      assertThrows(
          IllegalArgumentException.class,
          () -> new GPUSearchParams.Builder().withWriterThreads(v).build());
    }
  }

  @Test
  public void testGPUSearchParamsInvalidCagraGraphBuildAlgo() {
    assertThrows(
        IllegalArgumentException.class,
        () -> new GPUSearchParams.Builder().withCagraGraphBuildAlgo(null).build());
  }

  @Test
  public void testGPUSearchParamsInvalidIndexType() {
    assertThrows(
        IllegalArgumentException.class,
        () -> new GPUSearchParams.Builder().withIndexType(null).build());
  }

  @Test
  public void testGPUSearchParamsInvalidStrategy() {
    assertThrows(
        IllegalArgumentException.class,
        () -> new GPUSearchParams.Builder().withStrategy(null).build());
  }

  @Test
  public void testGPUSearchParamsInvalidCuvsDistanceType() {
    assertThrows(
        IllegalArgumentException.class,
        () -> new GPUSearchParams.Builder().withCuvsDistanceType(null).build());
  }

  @Test
  public void testGPUSearchParamsInvalidNNDescentNumIterations() {
    for (int v :
        new int[] {
          random.nextInt(MIN_VALUE, (int) MIN_NN_DESCENT_NUM_ITERATIONS),
          random.nextInt((int) (MAX_NN_DESCENT_NUM_ITERATIONS + 1), MAX_VALUE)
        }) {
      assertThrows(
          IllegalArgumentException.class,
          () -> new GPUSearchParams.Builder().withNNDescentNumIterations(v).build());
    }
  }

  @BeforeClass
  public static void beforeClass() {
    random = random();
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import com.carrotsearch.randomizedtesting.RandomizedRunner;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/**
 * A CAGRA graph needs at least two rows: with a single row there is no neighbor to link to, so
 * both graph degrees collapse to zero. Building one used to launch the kNN graph kernels anyway,
 * which hit an illegal memory access and aborted the JVM (exit code 134). It has to fail with an
 * exception instead.
 */
@RunWith(RandomizedRunner.class)
public class CagraSingleRowIT extends CuVSTestCase {

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxSupportedArch());
    initializeRandom();
  }

  private void assertSingleRowBuildFails(CagraGraphBuildAlgo algo) throws Throwable {
    float[][] vectors = generateData(random, 1, 16);
    try (CuVSResources resources = CheckedCuVSResources.create()) {
      CagraIndexParams indexParams =
          new CagraIndexParams.Builder()
              .withCagraGraphBuildAlgo(algo)
              .withGraphDegree(64)
              .withIntermediateGraphDegree(128)
              .withMetric(CuvsDistanceType.L2Expanded)
              .build();
      RuntimeException e =
          assertThrows(
              RuntimeException.class,
              () ->
                  CagraIndex.newBuilder(resources)
                      .withDataset(vectors)
                      .withIndexParams(indexParams)
                      .build());
      assertTrue(e.getMessage(), e.getMessage().contains("at least 2 rows"));
    }
  }

  @Test
  public void testSingleRowNnDescentBuildThrows() throws Throwable {
    assertSingleRowBuildFails(CagraGraphBuildAlgo.NN_DESCENT);
  }

  @Test
  public void testSingleRowIvfPqBuildThrows() throws Throwable {
    assertSingleRowBuildFails(CagraGraphBuildAlgo.IVF_PQ);
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.lucene.CuVS2510GPUVectorsWriter.IndexType;
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
    assertEquals(64, params.getGraphdegree());
    assertEquals(128, params.getIntermediateGraphDegree());
    assertEquals(1, params.getWriterThreads());
    assertEquals(CagraGraphBuildAlgo.NN_DESCENT, params.getCagraGraphBuildAlgo());
    assertEquals(IndexType.CAGRA, params.getIndexType());
  }

  @Test
  public void testGPUSearchParamsInvalidGraphDegree() {
    for (int v :
        new int[] {random.nextInt(Integer.MIN_VALUE, 1), random.nextInt(65, Integer.MAX_VALUE)}) {
      assertThrows(
          IllegalArgumentException.class,
          () -> new GPUSearchParams.Builder().withGraphDegree(v).build());
    }
  }

  @Test
  public void testGPUSearchParamsInvalidIntermediateGraphDegree() {
    for (int v :
        new int[] {random.nextInt(Integer.MIN_VALUE, 1), random.nextInt(129, Integer.MAX_VALUE)}) {
      assertThrows(
          IllegalArgumentException.class,
          () -> new GPUSearchParams.Builder().withIntermediateGraphDegree(v).build());
    }
  }

  @Test
  public void testGPUSearchParamsInvalidWriterThreads() {
    for (int v :
        new int[] {random.nextInt(Integer.MIN_VALUE, 1), random.nextInt(2, Integer.MAX_VALUE)}) {
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

  @BeforeClass
  public static void beforeClass() {
    random = random();
  }
}

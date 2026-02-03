/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

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
    assertEquals(16, params.getBeamWidth());
    assertEquals(64, params.getGraphdegree());
    assertEquals(2, params.getHnswLayers());
    assertEquals(128, params.getIntermediateGraphDegree());
    assertEquals(8, params.getMaxConn());
    assertEquals(1, params.getWriterThreads());
  }

  @Test
  public void testAcceleratedHNSWParamsInvalidBeamWidth() {
    for (int v :
        new int[] {random.nextInt(Integer.MIN_VALUE, 1), random.nextInt(17, Integer.MAX_VALUE)}) {
      assertThrows(
          IllegalArgumentException.class,
          () -> new AcceleratedHNSWParams.Builder().withBeamWidth(v).build());
    }
  }

  @Test
  public void testAcceleratedHNSWParamsInvalidGraphDegree() {
    for (int v :
        new int[] {random.nextInt(Integer.MIN_VALUE, 1), random.nextInt(65, Integer.MAX_VALUE)}) {
      assertThrows(
          IllegalArgumentException.class,
          () -> new AcceleratedHNSWParams.Builder().withGraphDegree(v).build());
    }
  }

  @Test
  public void testAcceleratedHNSWParamsInvalidHNSWLayers() {
    for (int v :
        new int[] {random.nextInt(Integer.MIN_VALUE, 1), random.nextInt(3, Integer.MAX_VALUE)}) {
      assertThrows(
          IllegalArgumentException.class,
          () -> new AcceleratedHNSWParams.Builder().withHNSWLayer(v).build());
    }
  }

  @Test
  public void testAcceleratedHNSWParamsInvalidIntGraphDegree() {
    for (int v :
        new int[] {random.nextInt(Integer.MIN_VALUE, 1), random.nextInt(129, Integer.MAX_VALUE)}) {
      assertThrows(
          IllegalArgumentException.class,
          () -> new AcceleratedHNSWParams.Builder().withIntermediateGraphDegree(v).build());
    }
  }

  @Test
  public void testAcceleratedHNSWParamsInvalidMaxConn() {
    for (int v :
        new int[] {random.nextInt(Integer.MIN_VALUE, 1), random.nextInt(9, Integer.MAX_VALUE)}) {
      assertThrows(
          IllegalArgumentException.class,
          () -> new AcceleratedHNSWParams.Builder().withMaxConn(v).build());
    }
  }

  @Test
  public void testAcceleratedHNSWParamsInvalidWriterThreads() {
    for (int v :
        new int[] {random.nextInt(Integer.MIN_VALUE, 1), random.nextInt(2, Integer.MAX_VALUE)}) {
      assertThrows(
          IllegalArgumentException.class,
          () -> new AcceleratedHNSWParams.Builder().withWriterThreads(v).build());
    }
  }

  @BeforeClass
  public static void beforeClass() {
    random = random();
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import java.util.Arrays;
import java.util.List;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

public class TestAcceleratedHNSWUtils extends LuceneTestCase {

  @Test
  public void testHnswMUsesCeilingAndNeverReturnsZero() {
    assertEquals(1, AcceleratedHNSWUtils.hnswM(0));
    assertEquals(1, AcceleratedHNSWUtils.hnswM(1));
    assertEquals(1, AcceleratedHNSWUtils.hnswM(2));
    assertEquals(2, AcceleratedHNSWUtils.hnswM(3));
    assertEquals(32, AcceleratedHNSWUtils.hnswM(63));
    assertEquals(32, AcceleratedHNSWUtils.hnswM(64));
    assertEquals(33, AcceleratedHNSWUtils.hnswM(65));
  }

  @Test
  public void testUpperLayerSizesUseCeilingAndStopNaturally() {
    List<int[]> layers = AcceleratedHNSWUtils.selectUpperLayerNodes(10_000, 99, 32, 44L);
    assertEquals(2, layers.size());
    assertEquals(313, layers.get(0).length);
    assertEquals(10, layers.get(1).length);

    List<int[]> capped = AcceleratedHNSWUtils.selectUpperLayerNodes(10_000, 2, 32, 44L);
    assertEquals(1, capped.size());
    assertEquals(313, capped.get(0).length);
  }

  @Test
  public void testUpperLayerSamplingIsDeterministicAndNested() {
    List<int[]> first = AcceleratedHNSWUtils.selectUpperLayerNodes(10_000, 99, 32, 987654L);
    List<int[]> repeat = AcceleratedHNSWUtils.selectUpperLayerNodes(10_000, 99, 32, 987654L);
    List<int[]> differentSeed = AcceleratedHNSWUtils.selectUpperLayerNodes(10_000, 99, 32, 987655L);

    assertEquals(first.size(), repeat.size());
    for (int i = 0; i < first.size(); i++) {
      assertArrayEquals(first.get(i), repeat.get(i));
    }
    assertFalse(Arrays.equals(first.get(0), differentSeed.get(0)));

    for (int node : first.get(1)) {
      assertTrue(Arrays.binarySearch(first.get(0), node) >= 0);
    }
  }

  @Test
  public void testUpperLayersRequireStrictShrink() {
    assertTrue(AcceleratedHNSWUtils.selectUpperLayerNodes(1_000, 99, 1, 44L).isEmpty());
    assertTrue(AcceleratedHNSWUtils.selectUpperLayerNodes(32, 99, 32, 44L).isEmpty());
    assertTrue(AcceleratedHNSWUtils.selectUpperLayerNodes(10_000, 1, 32, 44L).isEmpty());
  }
}

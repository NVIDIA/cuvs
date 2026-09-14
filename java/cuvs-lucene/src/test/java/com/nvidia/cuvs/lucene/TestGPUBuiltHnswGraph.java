/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

public class TestGPUBuiltHnswGraph extends LuceneTestCase {

  @Test
  public void findsUpperLayerOrdinalInSortedNodeIds() {
    int[] nodeIds = {3, 11, 27, 42, 81};

    assertEquals(0, GPUBuiltHnswGraph.findUpperLayerOrdinal(nodeIds, 3));
    assertEquals(2, GPUBuiltHnswGraph.findUpperLayerOrdinal(nodeIds, 27));
    assertEquals(4, GPUBuiltHnswGraph.findUpperLayerOrdinal(nodeIds, 81));
  }

  @Test
  public void returnsMissingOrdinalForUnknownUpperLayerNode() {
    int[] nodeIds = {3, 11, 27, 42, 81};

    assertEquals(-1, GPUBuiltHnswGraph.findUpperLayerOrdinal(nodeIds, 2));
    assertEquals(-1, GPUBuiltHnswGraph.findUpperLayerOrdinal(nodeIds, 28));
    assertEquals(-1, GPUBuiltHnswGraph.findUpperLayerOrdinal(nodeIds, 82));
  }

  @Test(timeout = 5000L)
  public void repeatedUpperLayerLookupsDoNotUseLinearScans() {
    int[] nodeIds = new int[1 << 20];
    for (int i = 0; i < nodeIds.length; i++) {
      nodeIds[i] = i * 2;
    }

    // These near-tail lookups require about 100 billion comparisons with a linear scan, while a
    // binary search performs about two million. The wide timeout keeps ordinary CI noise
    // irrelevant.
    long ordinalSum = 0;
    for (int i = 0; i < 100_000; i++) {
      int expectedOrdinal = nodeIds.length - 1 - (i & 1023);
      int ordinal = GPUBuiltHnswGraph.findUpperLayerOrdinal(nodeIds, nodeIds[expectedOrdinal]);
      assertEquals(expectedOrdinal, ordinal);
      ordinalSum += ordinal;
    }
    assertTrue(ordinalSum > 0);
  }
}

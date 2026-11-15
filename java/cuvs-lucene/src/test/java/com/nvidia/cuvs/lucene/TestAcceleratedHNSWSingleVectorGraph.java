/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

public class TestAcceleratedHNSWSingleVectorGraph extends LuceneTestCase {

  @Test
  public void testSingleVectorGraphHasNoNeighbors() throws Throwable {
    assumeTrue("cuVS not supported", ThreadLocalCuVSResourcesProvider.isSupported());

    GPUBuiltHnswGraph graph = AcceleratedHNSWUtils.createSingleVectorHnswGraph(1, 32);

    assertEquals(1, graph.numLevels());
    assertEquals(0, graph.maxConn());
    assertEquals(0, graph.getNeighbors(0, 0).size());
    graph.seek(0, 0);
    assertEquals(NO_MORE_DOCS, graph.nextNeighbor());
  }
}

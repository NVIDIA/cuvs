/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.QuantizationType;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.List;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

public class TestAcceleratedHNSWUpperLayers extends LuceneTestCase {

  @Test
  public void testLegacyListOverloadDescriptorIsPresent() throws Exception {
    Method method =
        AcceleratedHNSWUtils.class.getMethod(
            "createMultiLayerHnswGraph",
            FieldInfo.class,
            int.class,
            int.class,
            CuVSMatrix.class,
            List.class,
            int.class,
            CagraIndexParams.class,
            QuantizationType.class);

    assertEquals(GPUBuiltHnswGraph.class, method.getReturnType());

    Method matrixOverload =
        AcceleratedHNSWUtils.class.getDeclaredMethod(
            "createMultiLayerHnswGraph",
            int.class,
            CuVSMatrix.class,
            CuVSMatrix.class,
            int.class,
            CagraIndexParams.class,
            QuantizationType.class);
    assertFalse(Modifier.isPublic(matrixOverload.getModifiers()));
  }

  @Test
  public void testSingleVectorGraphHasNoNeighbors() throws Throwable {
    GPUBuiltHnswGraph graph = AcceleratedHNSWUtils.createSingleVectorHnswGraph(1, 32);

    assertEquals(1, graph.numLevels());
    assertEquals(0, graph.maxConn());
    assertEquals(0, graph.getNeighbors(0, 0).size());
    graph.seek(0, 0);
    assertEquals(NO_MORE_DOCS, graph.nextNeighbor());
  }
}

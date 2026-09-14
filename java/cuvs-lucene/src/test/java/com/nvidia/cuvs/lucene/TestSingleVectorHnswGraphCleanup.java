/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CuVSMatrix;
import java.lang.reflect.Proxy;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

public class TestSingleVectorHnswGraphCleanup extends LuceneTestCase {

  @Test
  public void testOwnedAdjacencyMatrixIsClosedAfterMaterialization() throws Throwable {
    AtomicBoolean rowRead = new AtomicBoolean();
    AtomicInteger closeCount = new AtomicInteger();
    CuVSMatrix matrix = matrix(rowRead, closeCount, null, null);

    GPUBuiltHnswGraph graph = AcceleratedHNSWUtils.createSingleVectorHnswGraph(1, 4, matrix);

    assertTrue("the adjacency row was not materialized", rowRead.get());
    assertEquals("the owned matrix must be closed exactly once", 1, closeCount.get());
    assertEquals(1, graph.size());
    assertEquals(1, graph.numLevels());
  }

  @Test
  public void testCloseFailureIsSuppressedOntoMaterializationFailure() {
    AtomicBoolean rowRead = new AtomicBoolean();
    AtomicInteger closeCount = new AtomicInteger();
    IllegalStateException materializationFailure =
        new IllegalStateException("materialization failed");
    IllegalStateException closeFailure = new IllegalStateException("matrix close failed");
    CuVSMatrix matrix = matrix(rowRead, closeCount, materializationFailure, closeFailure);

    IllegalStateException failure =
        expectThrows(
            IllegalStateException.class,
            () -> AcceleratedHNSWUtils.createSingleVectorHnswGraph(1, 4, matrix));

    assertSame(materializationFailure, failure);
    assertTrue("the failing adjacency row was not read", rowRead.get());
    assertEquals("the failing owned matrix must be closed exactly once", 1, closeCount.get());
    assertArrayEquals(new Throwable[] {closeFailure}, failure.getSuppressed());
  }

  private static CuVSMatrix matrix(
      AtomicBoolean rowRead,
      AtomicInteger closeCount,
      RuntimeException materializationFailure,
      RuntimeException closeFailure) {
    return (CuVSMatrix)
        Proxy.newProxyInstance(
            CuVSMatrix.class.getClassLoader(),
            new Class<?>[] {CuVSMatrix.class},
            (proxy, method, arguments) -> {
              if (method.getName().equals("getRow")) {
                rowRead.set(true);
                if (materializationFailure != null) {
                  throw materializationFailure;
                }
                return null;
              }
              if (method.getName().equals("close")) {
                closeCount.incrementAndGet();
                if (closeFailure != null) {
                  throw closeFailure;
                }
                return null;
              }
              throw new UnsupportedOperationException(method.getName());
            });
  }
}

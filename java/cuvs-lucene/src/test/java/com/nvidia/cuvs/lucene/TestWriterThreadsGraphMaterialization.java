/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CuVSDeviceMatrix;
import com.nvidia.cuvs.CuVSHostMatrix;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.QuantizationType;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.hnsw.HnswGraph.NodesIterator;
import org.apache.lucene.util.hnsw.NeighborArray;
import org.junit.Test;

public class TestWriterThreadsGraphMaterialization extends LuceneTestCase {

  private static final int NUM_NODES = GPUBuiltHnswGraph.PARALLEL_MIN_NODES + 1_000;
  private static final int DEGREE = 12;

  @Test
  public void testParallelMaterializationMatchesSerial() throws Exception {
    int[][] adjacency = randomAdjacency(NUM_NODES, DEGREE, new Random(1));

    try (CuVSMatrix matrix = new IntGraphTestMatrix(adjacency)) {
      GPUBuiltHnswGraph serial = newSingleLayerGraph(matrix, 1);
      GPUBuiltHnswGraph parallel = newSingleLayerGraph(matrix, 4);
      assertGraphsEqual(serial, parallel);
    }
  }

  @Test
  public void testGraphCopyBudgetHandlesLargeShapesWithoutOverflow() {
    assertTrue(GPUBuiltHnswGraph.fitsParallelGraphCopyBudget(25_000_000L, 32));
    assertFalse(GPUBuiltHnswGraph.fitsParallelGraphCopyBudget(100_000_000L, 32));
    assertFalse(GPUBuiltHnswGraph.fitsParallelGraphCopyBudget(Long.MAX_VALUE, Long.MAX_VALUE));
    assertFalse(GPUBuiltHnswGraph.fitsParallelGraphCopyBudget(-1, 32));
  }

  @Test
  public void testOversizedDeviceGraphUsesSerialFallback() throws Exception {
    int[][] adjacency = randomAdjacency(NUM_NODES, 1, new Random(2));
    long oversizedColumns =
        GPUBuiltHnswGraph.MAX_PARALLEL_GRAPH_COPY_BYTES / Integer.BYTES / NUM_NODES + 1;

    try (CuVSMatrix matrix = new IntGraphTestMatrix.Device(adjacency, oversizedColumns)) {
      GPUBuiltHnswGraph graph = newSingleLayerGraph(matrix, 4);
      assertEquals(NUM_NODES, graph.size());
    }
  }

  @Test
  public void testFailedDeviceCopyClosesHostAllocationAndSuppressesCloseFailure() {
    RuntimeException copyFailure = new RuntimeException("copy failed");
    RuntimeException closeFailure = new RuntimeException("close failed");
    AtomicInteger hostCloseCount = new AtomicInteger();
    CuVSDeviceMatrix source =
        new IntGraphTestMatrix.Device(new int[][] {{0}}, 1) {
          @Override
          public void toHost(CuVSHostMatrix target) {
            throw copyFailure;
          }
        };
    CuVSHostMatrix hostCopy = new TrackingHostMatrix(hostCloseCount, closeFailure);

    RuntimeException thrown =
        assertThrows(
            RuntimeException.class, () -> GPUBuiltHnswGraph.copyToHost(source, () -> hostCopy));

    assertSame(copyFailure, thrown);
    assertEquals(1, hostCloseCount.get());
    assertArrayEquals(new Throwable[] {closeFailure}, thrown.getSuppressed());
  }

  @Test
  public void testSuccessfulDeviceCopyTransfersHostOwnershipToCaller() {
    AtomicInteger hostCloseCount = new AtomicInteger();
    CuVSHostMatrix hostCopy = new TrackingHostMatrix(hostCloseCount, null);
    CuVSDeviceMatrix source =
        new IntGraphTestMatrix.Device(new int[][] {{0}}, 1) {
          @Override
          public void toHost(CuVSHostMatrix target) {
            assertSame(hostCopy, target);
          }
        };

    CuVSHostMatrix returned = GPUBuiltHnswGraph.copyToHost(source, () -> hostCopy);
    assertSame(hostCopy, returned);
    assertEquals(0, hostCloseCount.get());
    returned.close();
    assertEquals(1, hostCloseCount.get());
  }

  @Test
  public void testLegacyPublicDescriptorsRemainAvailable() throws Exception {
    Constructor<GPUBuiltHnswGraph> constructor =
        GPUBuiltHnswGraph.class.getConstructor(int.class, int.class, List.class, List.class);
    assertEquals(0, constructor.getExceptionTypes().length);

    Method createGraph =
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
    assertEquals(GPUBuiltHnswGraph.class, createGraph.getReturnType());

    Method writeGraph =
        AcceleratedHNSWUtils.class.getMethod(
            "writeGraph", GPUBuiltHnswGraph.class, IndexOutput.class);
    assertArrayEquals(new Class<?>[] {IOException.class}, writeGraph.getExceptionTypes());
  }

  private static GPUBuiltHnswGraph newSingleLayerGraph(CuVSMatrix adjacency, int workers)
      throws Exception {
    return GPUBuiltHnswGraph.create(
        NUM_NODES,
        /* dimensions= */ 4,
        Arrays.asList((int[]) null),
        List.of(adjacency),
        workers,
        4);
  }

  private static void assertGraphsEqual(GPUBuiltHnswGraph expected, GPUBuiltHnswGraph actual) {
    assertEquals(expected.numLevels(), actual.numLevels());
    for (int level = 0; level < expected.numLevels(); level++) {
      int[] nodes = NodesIterator.getSortedNodes(expected.getNodesOnLevel(level));
      for (int node : nodes) {
        NeighborArray expectedNeighbors = expected.getNeighbors(level, node);
        NeighborArray actualNeighbors = actual.getNeighbors(level, node);
        assertEquals(expectedNeighbors.size(), actualNeighbors.size());
        assertArrayEquals(
            "node " + node + " on level " + level,
            Arrays.copyOf(expectedNeighbors.nodes(), expectedNeighbors.size()),
            Arrays.copyOf(actualNeighbors.nodes(), actualNeighbors.size()));
      }
    }
  }

  private static int[][] randomAdjacency(int nodes, int degree, Random random) {
    int[][] adjacency = new int[nodes][degree];
    for (int[] row : adjacency) {
      for (int index = 0; index < row.length; index++) {
        row[index] = random.nextInt(nodes);
      }
    }
    return adjacency;
  }

  private static final class TrackingHostMatrix extends IntGraphTestMatrix
      implements CuVSHostMatrix {
    private final AtomicInteger closeCount;
    private final RuntimeException closeFailure;

    TrackingHostMatrix(AtomicInteger closeCount, RuntimeException closeFailure) {
      super(new int[][] {{0}});
      this.closeCount = closeCount;
      this.closeFailure = closeFailure;
    }

    @Override
    public int get(int row, int column) {
      return 0;
    }

    @Override
    public void close() {
      closeCount.incrementAndGet();
      if (closeFailure != null) {
        throw closeFailure;
      }
    }
  }
}

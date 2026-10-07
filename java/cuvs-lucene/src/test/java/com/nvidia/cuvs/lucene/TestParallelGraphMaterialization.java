/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import com.nvidia.cuvs.CuVSDeviceMatrix;
import com.nvidia.cuvs.CuVSHostMatrix;
import com.nvidia.cuvs.CuVSMatrix;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.HnswGraph.NodesIterator;
import org.apache.lucene.util.hnsw.NeighborArray;
import org.junit.Test;

/** Verifies serial and parallel CAGRA-adjacency materialization and host-copy ownership. */
public class TestParallelGraphMaterialization extends LuceneTestCase {

  private static final int NUM_NODES = GPUBuiltHnswGraph.PARALLEL_MIN_NODES + 1000;
  private static final int DEGREE = 12;
  private static final int GRAPH_THREADS = 4;

  @Test
  public void parallelMaterializationMatchesSerial() throws Exception {
    int[][] rows = IntGraphTestMatrix.randomRows(NUM_NODES, DEGREE, 1);
    IntGraphTestMatrix.ParallelExecutionProbe executionProbe =
        new IntGraphTestMatrix.ParallelExecutionProbe();
    try (CuVSMatrix serialMatrix = new IntGraphTestMatrix(rows);
        CuVSMatrix parallelMatrix =
            new IntGraphTestMatrix.TrackingHostMatrix(
                rows, new AtomicInteger(), null, executionProbe)) {
      GPUBuiltHnswGraph serial =
          new GPUBuiltHnswGraph(
              NUM_NODES, /* dimensions= */ 4, Arrays.asList((int[]) null), List.of(serialMatrix));
      GPUBuiltHnswGraph parallel = newSingleLayerGraph(parallelMatrix, GRAPH_THREADS);
      assertGraphsEqual(serial, parallel);
      assertTrue(executionProbe.threadCount() > 1);
    }
  }

  @Test
  public void overflowingDeviceShapeUsesSerialFallback() throws Exception {
    int[][] adjacency = IntGraphTestMatrix.randomRows(NUM_NODES, 1, 0);
    try (CuVSMatrix matrix = new IntGraphTestMatrix.DeviceMatrix(adjacency, Long.MAX_VALUE)) {
      GPUBuiltHnswGraph graph = newSingleLayerGraph(matrix, GRAPH_THREADS);
      assertEquals(NUM_NODES, graph.size());
    }
  }

  @Test
  public void failedDeviceCopyClosesHostAllocationAndSuppressesCloseFailure() {
    RuntimeException copyFailure = new RuntimeException("copy failed");
    RuntimeException closeFailure = new RuntimeException("close failed");
    AtomicInteger hostCloseCount = new AtomicInteger();
    CuVSDeviceMatrix source =
        new IntGraphTestMatrix.DeviceMatrix(new int[][] {{0}}, 1) {
          @Override
          public void toHost(CuVSHostMatrix target) {
            throw copyFailure;
          }
        };
    CuVSHostMatrix hostCopy =
        new IntGraphTestMatrix.TrackingHostMatrix(hostCloseCount, closeFailure);

    RuntimeException thrown =
        assertThrows(
            RuntimeException.class, () -> GPUBuiltHnswGraph.copyToHost(source, () -> hostCopy));

    assertSame(copyFailure, thrown);
    assertEquals(1, hostCloseCount.get());
    assertArrayEquals(new Throwable[] {closeFailure}, thrown.getSuppressed());
  }

  @Test
  public void admittedDeviceCopyIsMaterializedInParallelAndReleased() throws Exception {
    int[][] expectedRows = IntGraphTestMatrix.randomRows(NUM_NODES, DEGREE, 3);
    int[][] wrongSourceRows = new int[NUM_NODES][DEGREE];
    AtomicInteger hostCloseCount = new AtomicInteger();
    AtomicInteger copyCount = new AtomicInteger();
    IntGraphTestMatrix.ParallelExecutionProbe executionProbe =
        new IntGraphTestMatrix.ParallelExecutionProbe();
    CuVSHostMatrix hostCopy =
        new IntGraphTestMatrix.TrackingHostMatrix(
            expectedRows, hostCloseCount, null, executionProbe);
    CuVSDeviceMatrix source =
        new IntGraphTestMatrix.DeviceMatrix(wrongSourceRows, DEGREE) {
          @Override
          public void toHost(CuVSHostMatrix target) {
            assertSame(hostCopy, target);
            copyCount.incrementAndGet();
          }
        };
    long requiredCopyBytes = GraphCopyMemoryBudget.requiredCopyBytes(NUM_NODES, DEGREE);
    GraphCopyMemoryBudget budget = new GraphCopyMemoryBudget();

    NeighborArray[] neighbors =
        GPUBuiltHnswGraph.materializeDeviceAdjacency(
            source, NUM_NODES, GRAPH_THREADS, budget, requiredCopyBytes, () -> hostCopy);

    for (int node = 0; node < NUM_NODES; node++) {
      assertArrayEquals(
          expectedRows[node], Arrays.copyOf(neighbors[node].nodes(), neighbors[node].size()));
    }
    assertEquals(1, copyCount.get());
    assertEquals(1, hostCloseCount.get());
    assertTrue(executionProbe.threadCount() > 1);
    try (GraphCopyMemoryBudget.Reservation ignored =
        budget.tryReserve(NUM_NODES, DEGREE, requiredCopyBytes).orElseThrow()) {
      // The first reservation was released after materialization.
    }
  }

  @Test
  public void configuredBudgetControlsWhetherDeviceCopyRuns() throws Exception {
    int[][] sourceRows = IntGraphTestMatrix.randomRows(NUM_NODES, DEGREE, 5);
    AtomicInteger copyCount = new AtomicInteger();
    CuVSDeviceMatrix source =
        new IntGraphTestMatrix.DeviceMatrix(sourceRows, DEGREE) {
          @Override
          public void toHost(CuVSHostMatrix target) {
            copyCount.incrementAndGet();
          }
        };
    long requiredCopyBytes = GraphCopyMemoryBudget.requiredCopyBytes(NUM_NODES, DEGREE);
    GraphCopyMemoryBudget budget = new GraphCopyMemoryBudget();

    GPUBuiltHnswGraph.materializeDeviceAdjacency(
        source,
        NUM_NODES,
        GRAPH_THREADS,
        budget,
        requiredCopyBytes - 1,
        () -> new IntGraphTestMatrix.TrackingHostMatrix(new AtomicInteger(), null));
    assertEquals(0, copyCount.get());

    CuVSHostMatrix hostCopy =
        new IntGraphTestMatrix.TrackingHostMatrix(sourceRows, new AtomicInteger(), null, null);
    GPUBuiltHnswGraph.materializeDeviceAdjacency(
        source, NUM_NODES, GRAPH_THREADS, budget, requiredCopyBytes, () -> hostCopy);
    assertEquals(1, copyCount.get());
  }

  private static GPUBuiltHnswGraph newSingleLayerGraph(CuVSMatrix layer0Adjacency, int numThreads)
      throws IOException {
    return new GPUBuiltHnswGraph(
        NUM_NODES,
        /* dimensions= */ 4,
        Arrays.asList((int[]) null),
        List.of(layer0Adjacency),
        numThreads);
  }

  private static void assertGraphsEqual(HnswGraph a, HnswGraph b) throws Exception {
    assertEquals(a.numLevels(), b.numLevels());
    for (int level = 0; level < a.numLevels(); level++) {
      int[] nodes = NodesIterator.getSortedNodes(a.getNodesOnLevel(level));
      for (int node : nodes) {
        assertArrayEquals(
            "node " + node + " at level " + level + " has different neighbors",
            arcsOf(a, level, node),
            arcsOf(b, level, node));
      }
    }
  }

  private static int[] arcsOf(HnswGraph graph, int level, int node) throws Exception {
    graph.seek(level, node);
    List<Integer> arcs = new ArrayList<>();
    for (int n = graph.nextNeighbor(); n != NO_MORE_DOCS; n = graph.nextNeighbor()) {
      arcs.add(n);
    }
    return arcs.stream().mapToInt(Integer::intValue).toArray();
  }
}

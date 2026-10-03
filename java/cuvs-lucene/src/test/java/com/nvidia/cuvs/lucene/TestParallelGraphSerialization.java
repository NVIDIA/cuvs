/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CuVSMatrix;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.hnsw.NeighborArray;
import org.junit.Test;

/** Verifies parallel level-zero graph serialization is byte-identical to serial serialization. */
public class TestParallelGraphSerialization extends LuceneTestCase {

  private static final int NUM_NODES = GPUBuiltHnswGraph.PARALLEL_MIN_NODES + 1000;
  private static final int DEGREE = 12;
  private static final int GRAPH_THREADS = 4;

  @Test
  public void parallelSerializationMatchesSerial() throws Exception {
    try (CuVSMatrix matrix = IntGraphTestMatrix.random(NUM_NODES, DEGREE, 2);
        Directory dir = new ByteBuffersDirectory()) {
      GPUBuiltHnswGraph serialGraph = newSingleLayerGraph(matrix);
      IntGraphTestMatrix.ParallelExecutionProbe executionProbe =
          new IntGraphTestMatrix.ParallelExecutionProbe();
      GPUBuiltHnswGraph parallelGraph = new RecordingGraph(matrix, executionProbe);
      assertSerialAndParallelMatch(serialGraph, parallelGraph, dir);
      assertTrue(executionProbe.threadCount() > 1);
    }
  }

  @Test
  public void parallelSerializationMatchesSerialAcrossByteBoundedWave() throws Exception {
    int maxConn = 512;
    int waveNodes = AcceleratedHNSWUtils.serializationWaveNodes(maxConn);
    int numNodes = Math.max(GPUBuiltHnswGraph.PARALLEL_MIN_NODES + 1, waveNodes + 2);
    long worstCaseNodeBytes = 5L + maxConn * 5L;
    IntGraphTestMatrix.ParallelExecutionProbe executionProbe =
        new IntGraphTestMatrix.ParallelExecutionProbe();

    try (Directory dir = new ByteBuffersDirectory()) {
      assertTrue(waveNodes < AcceleratedHNSWUtils.MAX_SERIALIZATION_WAVE_NODES);
      assertTrue(
          waveNodes * worstCaseNodeBytes <= AcceleratedHNSWUtils.MAX_SERIALIZED_BYTES_PER_WAVE);
      assertTrue(
          (waveNodes + 1L) * worstCaseNodeBytes
              > AcceleratedHNSWUtils.MAX_SERIALIZED_BYTES_PER_WAVE);
      assertSerialAndParallelMatch(
          new LazyBoundaryGraph(numNodes, maxConn, waveNodes, null),
          new LazyBoundaryGraph(numNodes, maxConn, waveNodes, executionProbe),
          dir);
      assertTrue(executionProbe.threadCount() > 1);
    }
  }

  @Test
  public void lowDegreeSerializationWaveUsesAbsoluteNodeCap() {
    assertEquals(
        AcceleratedHNSWUtils.MAX_SERIALIZATION_WAVE_NODES,
        AcceleratedHNSWUtils.serializationWaveNodes(/* maxConn= */ 0));
  }

  @Test
  public void serialSerializationRejectsMissingAdjacency() throws Exception {
    assertMissingAdjacencyRejected(/* graphSize= */ 1, /* graphThreads= */ 1);
  }

  @Test
  public void parallelSerializationRejectsMissingAdjacency() throws Exception {
    assertMissingAdjacencyRejected(
        GPUBuiltHnswGraph.PARALLEL_MIN_NODES, /* graphThreads= */ GRAPH_THREADS);
  }

  private static void assertMissingAdjacencyRejected(int graphSize, int graphThreads)
      throws Exception {
    try (Directory dir = new ByteBuffersDirectory();
        IndexOutput out = dir.createOutput("missing-adjacency", IOContext.DEFAULT)) {
      GPUBuiltHnswGraph graph = new MissingAdjacencyGraph(graphSize);
      expectThrows(
          NullPointerException.class,
          () -> AcceleratedHNSWUtils.writeGraph(graph, out, graphThreads));
    }
  }

  private static void assertSerialAndParallelMatch(
      GPUBuiltHnswGraph serialGraph, GPUBuiltHnswGraph parallelGraph, Directory dir)
      throws Exception {
    int[][] serialOffsets;
    try (IndexOutput out = dir.createOutput("serial", IOContext.DEFAULT)) {
      serialOffsets = AcceleratedHNSWUtils.writeGraph(serialGraph, out);
    }
    int[][] parallelOffsets;
    try (IndexOutput out = dir.createOutput("parallel", IOContext.DEFAULT)) {
      parallelOffsets = AcceleratedHNSWUtils.writeGraph(parallelGraph, out, GRAPH_THREADS);
    }

    assertEquals(serialOffsets.length, parallelOffsets.length);
    for (int level = 0; level < serialOffsets.length; level++) {
      assertArrayEquals(serialOffsets[level], parallelOffsets[level]);
    }
    assertArrayEquals(readAllBytes(dir, "serial"), readAllBytes(dir, "parallel"));
  }

  private static GPUBuiltHnswGraph newSingleLayerGraph(CuVSMatrix layer0Adjacency) {
    return new GPUBuiltHnswGraph(
        NUM_NODES, /* dimensions= */ 4, Arrays.asList((int[]) null), List.of(layer0Adjacency));
  }

  private static byte[] readAllBytes(Directory dir, String name) throws Exception {
    try (IndexInput in = dir.openInput(name, IOContext.DEFAULT)) {
      byte[] bytes = new byte[(int) in.length()];
      in.readBytes(bytes, 0, bytes.length);
      return bytes;
    }
  }

  private static final class RecordingGraph extends GPUBuiltHnswGraph {
    private final IntGraphTestMatrix.ParallelExecutionProbe executionProbe;

    RecordingGraph(
        CuVSMatrix layer0Adjacency, IntGraphTestMatrix.ParallelExecutionProbe executionProbe) {
      super(NUM_NODES, /* dimensions= */ 4, Arrays.asList((int[]) null), List.of(layer0Adjacency));
      this.executionProbe = executionProbe;
    }

    @Override
    public NeighborArray getNeighbors(int level, int node) {
      executionProbe.recordExecution();
      return super.getNeighbors(level, node);
    }
  }

  /**
   * Crosses a maxConn-derived byte-bounded wave with sparse rows and no retained heap graph.
   */
  private static final class LazyBoundaryGraph extends GPUBuiltHnswGraph {
    private static final NeighborArray EMPTY_NEIGHBORS = new NeighborArray(0, true);

    private final int graphSize;
    private final int maxConn;
    private final int waveNodes;
    private final IntGraphTestMatrix.ParallelExecutionProbe executionProbe;

    LazyBoundaryGraph(
        int graphSize,
        int maxConn,
        int waveNodes,
        IntGraphTestMatrix.ParallelExecutionProbe executionProbe)
        throws IOException {
      super(
          0,
          /* dimensions= */ 4,
          Arrays.asList((int[]) null),
          List.of(new IntGraphTestMatrix(new int[0][])),
          1);
      this.graphSize = graphSize;
      this.maxConn = maxConn;
      this.waveNodes = waveNodes;
      this.executionProbe = executionProbe;
    }

    @Override
    public int size() {
      return graphSize;
    }

    @Override
    public int maxConn() {
      return maxConn;
    }

    @Override
    public NodesIterator getNodesOnLevel(int level) {
      return new RangeNodesIterator(level == 0 ? graphSize : 0);
    }

    @Override
    public NeighborArray getNeighbors(int level, int node) {
      if (executionProbe != null) {
        executionProbe.recordExecution();
      }
      if (node < waveNodes - 1) {
        return EMPTY_NEIGHBORS;
      }
      NeighborArray neighbors = new NeighborArray(1, true);
      neighbors.addInOrder(node, 1.0f);
      return neighbors;
    }
  }

  private static final class MissingAdjacencyGraph extends GPUBuiltHnswGraph {
    private final int graphSize;

    MissingAdjacencyGraph(int graphSize) throws IOException {
      super(
          0,
          /* dimensions= */ 4,
          Arrays.asList((int[]) null),
          List.of(new IntGraphTestMatrix(new int[0][])),
          1);
      this.graphSize = graphSize;
    }

    @Override
    public int size() {
      return graphSize;
    }

    @Override
    public int maxConn() {
      return 0;
    }

    @Override
    public NodesIterator getNodesOnLevel(int level) {
      return new RangeNodesIterator(level == 0 ? graphSize : 0);
    }

    @Override
    public NeighborArray getNeighbors(int level, int node) {
      return null;
    }
  }

  private static final class RangeNodesIterator extends GPUBuiltHnswGraph.NodesIterator {
    private int current = -1;

    RangeNodesIterator(int size) {
      super(size);
    }

    @Override
    public boolean hasNext() {
      return current + 1 < size;
    }

    @Override
    public int nextInt() {
      return ++current;
    }

    @Override
    public int consume(int[] dest) {
      int count = Math.min(dest.length, size - (current + 1));
      for (int i = 0; i < count; i++) {
        dest[i] = ++current;
      }
      return count;
    }
  }
}

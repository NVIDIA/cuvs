/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CuVSMatrix;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.hnsw.NeighborArray;
import org.junit.Test;

public class TestWriterThreadsGraphSerialization extends LuceneTestCase {

  private static final int NUM_NODES = AcceleratedHNSWUtils.PARALLEL_MIN_NODES + 1_000;
  private static final int DEGREE = 12;
  private static final int REPORTED_MAX_CONN = 512;

  @Test
  public void testParallelSerializationMatchesSerialAcrossWaves() throws Exception {
    int[][] adjacency = randomAdjacency(NUM_NODES, DEGREE, new Random(2));

    try (CuVSMatrix matrix = new IntGraphTestMatrix(adjacency);
        Directory directory = new ByteBuffersDirectory()) {
      GPUBuiltHnswGraph serialGraph = new ReportedMaxConnGraph(matrix);
      GPUBuiltHnswGraph parallelGraph = new ReportedMaxConnGraph(matrix);

      int[][] serialOffsets;
      try (IndexOutput output = directory.createOutput("serial", IOContext.DEFAULT)) {
        serialOffsets = AcceleratedHNSWUtils.writeGraph(serialGraph, output, 1, 4);
      }
      int[][] parallelOffsets;
      try (IndexOutput output = directory.createOutput("parallel", IOContext.DEFAULT)) {
        parallelOffsets = AcceleratedHNSWUtils.writeGraph(parallelGraph, output, 4, 4);
      }

      assertEquals(serialOffsets.length, parallelOffsets.length);
      for (int level = 0; level < serialOffsets.length; level++) {
        assertArrayEquals(serialOffsets[level], parallelOffsets[level]);
      }
      assertArrayEquals(readAllBytes(directory, "serial"), readAllBytes(directory, "parallel"));
      assertTrue(
          "test must cross a bounded-wave boundary",
          AcceleratedHNSWUtils.nodesPerSerializationWave(REPORTED_MAX_CONN) < NUM_NODES);
    }
  }

  @Test
  public void testNodeEncodingHasStableLiteralBytes() throws Exception {
    NeighborArray neighbors = new NeighborArray(3, true);
    neighbors.addInOrder(130, 1.0f);
    neighbors.addInOrder(0, 0.9f);
    neighbors.addInOrder(128, 0.8f);

    SerializedGraph encoded = serialize(new LiteralGraph(131, neighbors));
    assertArrayEquals(new byte[] {3, 0, (byte) 0x80, 1, 2}, encoded.bytes());
    assertArrayEquals(new int[] {5}, encoded.offsets()[0]);

    SerializedGraph empty = serialize(new LiteralGraph(1, new NeighborArray(0, true)));
    assertArrayEquals(new byte[] {0}, empty.bytes());
    assertArrayEquals(new int[] {1}, empty.offsets()[0]);
  }

  @Test
  public void testSerializationWaveHonorsByteBudget() {
    for (int maxConn : new int[] {0, 1, 32, 88, 152, 512, Integer.MAX_VALUE}) {
      int nodes = AcceleratedHNSWUtils.nodesPerSerializationWave(maxConn);
      long maximumBytesPerNode = (Math.max(0L, maxConn) + 1L) * 5L;
      assertTrue(nodes > 0);
      if (maximumBytesPerNode > AcceleratedHNSWUtils.MAX_PARALLEL_ENCODE_BYTES) {
        assertEquals(1, nodes);
      } else {
        assertTrue(
            (long) nodes * maximumBytesPerNode <= AcceleratedHNSWUtils.MAX_PARALLEL_ENCODE_BYTES);
      }
    }
  }

  @Test
  public void testSerialSerializationRejectsMissingNeighborsWithoutWritingBytes() throws Exception {
    try (Directory directory = new ByteBuffersDirectory();
        IndexOutput output = directory.createOutput("missing", IOContext.DEFAULT)) {
      IOException thrown =
          assertThrows(
              IOException.class,
              () -> AcceleratedHNSWUtils.writeGraph(new MissingNeighborsGraph(), output));

      assertTrue(thrown.getMessage().contains("node 0 on level 0"));
      assertEquals(0L, output.getFilePointer());
    }
  }

  @Test
  public void testParallelSerializationRejectsMissingNeighborsWithoutWritingBytes()
      throws Exception {
    int[][] adjacency = randomAdjacency(NUM_NODES, DEGREE, new Random(3));
    try (CuVSMatrix matrix = new IntGraphTestMatrix(adjacency);
        Directory directory = new ByteBuffersDirectory();
        IndexOutput output = directory.createOutput("missing", IOContext.DEFAULT)) {
      GPUBuiltHnswGraph graph = new MissingParallelNeighborsGraph(matrix);
      IOException thrown =
          assertThrows(
              IOException.class, () -> AcceleratedHNSWUtils.writeGraph(graph, output, 4, 4));

      assertTrue(thrown.getMessage().contains("node " + (NUM_NODES / 2) + " on level 0"));
      assertEquals(0L, output.getFilePointer());
    }
  }

  private static SerializedGraph serialize(GPUBuiltHnswGraph graph) throws Exception {
    try (Directory directory = new ByteBuffersDirectory()) {
      int[][] offsets;
      try (IndexOutput output = directory.createOutput("encoded", IOContext.DEFAULT)) {
        offsets = AcceleratedHNSWUtils.writeGraph(graph, output);
      }
      return new SerializedGraph(readAllBytes(directory, "encoded"), offsets);
    }
  }

  private static byte[] readAllBytes(Directory directory, String file) throws Exception {
    try (IndexInput input = directory.openInput(file, IOContext.DEFAULT)) {
      byte[] bytes = new byte[Math.toIntExact(input.length())];
      input.readBytes(bytes, 0, bytes.length);
      return bytes;
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

  private static final class ReportedMaxConnGraph extends GPUBuiltHnswGraph {
    ReportedMaxConnGraph(CuVSMatrix adjacency) {
      super(NUM_NODES, /* dimensions= */ 4, Arrays.asList((int[]) null), List.of(adjacency));
    }

    @Override
    public int maxConn() {
      return REPORTED_MAX_CONN;
    }
  }

  private static final class LiteralGraph extends GPUBuiltHnswGraph {
    private final int graphSize;
    private final NeighborArray neighbors;

    LiteralGraph(int graphSize, NeighborArray neighbors) {
      super(
          1,
          /* dimensions= */ 1,
          Arrays.asList((int[]) null),
          List.of(new IntGraphTestMatrix(new int[][] {{}})));
      this.graphSize = graphSize;
      this.neighbors = neighbors;
    }

    @Override
    public int size() {
      return graphSize;
    }

    @Override
    public int maxConn() {
      return neighbors.size();
    }

    @Override
    public NeighborArray getNeighbors(int level, int node) {
      return neighbors;
    }
  }

  private static final class MissingNeighborsGraph extends GPUBuiltHnswGraph {
    MissingNeighborsGraph() {
      super(
          1,
          /* dimensions= */ 1,
          Arrays.asList((int[]) null),
          List.of(new IntGraphTestMatrix(new int[][] {{}})));
    }

    @Override
    public NeighborArray getNeighbors(int level, int node) {
      return null;
    }
  }

  private static final class MissingParallelNeighborsGraph extends GPUBuiltHnswGraph {
    MissingParallelNeighborsGraph(CuVSMatrix adjacency) {
      super(NUM_NODES, /* dimensions= */ 4, Arrays.asList((int[]) null), List.of(adjacency));
    }

    @Override
    public NeighborArray getNeighbors(int level, int node) {
      return node == NUM_NODES / 2 ? null : super.getNeighbors(level, node);
    }
  }

  private record SerializedGraph(byte[] bytes, int[][] offsets) {}
}

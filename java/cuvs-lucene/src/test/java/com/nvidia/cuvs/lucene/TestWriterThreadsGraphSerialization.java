/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CuVSDeviceMatrix;
import com.nvidia.cuvs.CuVSHostMatrix;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.RowView;
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

/** Verifies parallel level-zero graph serialization is byte-identical to serial serialization. */
public class TestWriterThreadsGraphSerialization extends LuceneTestCase {

  private static final int NUM_NODES = AcceleratedHNSWUtils.PARALLEL_MIN_NODES + 1000;
  private static final int DEGREE = 12;
  private static final int NUM_THREADS = 4;

  @Test
  public void parallelSerializationMatchesSerial() throws Exception {
    int[][] adjacency = randomAdjacency(NUM_NODES, DEGREE, new Random(2));

    try (CuVSMatrix matrix = new ArrayMatrix(adjacency);
        Directory dir = new ByteBuffersDirectory()) {
      GPUBuiltHnswGraph serialGraph = newSingleLayerGraph(matrix);
      GPUBuiltHnswGraph parallelGraph = newSingleLayerGraph(matrix);
      assertSerialAndParallelMatch(serialGraph, parallelGraph, dir);
    }
  }

  @Test
  public void parallelSerializationMatchesSerialAcrossFixedWaveBoundary() throws Exception {
    int numNodes = AcceleratedHNSWUtils.SERIALIZATION_WAVE_NODES + 1;
    assertTrue(numNodes > AcceleratedHNSWUtils.SERIALIZATION_WAVE_NODES);

    try (Directory dir = new ByteBuffersDirectory()) {
      assertSerialAndParallelMatch(new LazyEmptyGraph(numNodes), new LazyEmptyGraph(numNodes), dir);
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
      parallelOffsets = AcceleratedHNSWUtils.writeGraph(parallelGraph, out, NUM_THREADS);
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

  private static int[][] randomAdjacency(int numNodes, int degree, Random random) {
    int[][] adjacency = new int[numNodes][degree];
    for (int[] row : adjacency) {
      for (int j = 0; j < degree; j++) {
        row[j] = random.nextInt(numNodes);
      }
    }
    return adjacency;
  }

  /** Supplies an empty graph lazily so the fixed wave boundary can be tested with little heap. */
  private static final class LazyEmptyGraph extends GPUBuiltHnswGraph {
    private final int graphSize;

    LazyEmptyGraph(int graphSize) throws IOException {
      super(
          0,
          /* dimensions= */ 4,
          Arrays.asList((int[]) null),
          List.of(new ArrayMatrix(new int[0][])),
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

  private static final class ArrayMatrix implements CuVSMatrix {
    private final int[][] rows;

    ArrayMatrix(int[][] rows) {
      this.rows = rows;
    }

    @Override
    public long size() {
      return rows.length;
    }

    @Override
    public long columns() {
      return rows.length == 0 ? 0 : rows[0].length;
    }

    @Override
    public DataType dataType() {
      return DataType.INT;
    }

    @Override
    public RowView getRow(long row) {
      int[] values = rows[Math.toIntExact(row)];
      return new RowView() {
        @Override
        public long size() {
          return values.length;
        }

        @Override
        public int getAsInt(long index) {
          return values[Math.toIntExact(index)];
        }

        @Override
        public float getAsFloat(long index) {
          throw new UnsupportedOperationException();
        }

        @Override
        public byte getAsByte(long index) {
          throw new UnsupportedOperationException();
        }

        @Override
        public void toArray(int[] target) {
          System.arraycopy(values, 0, target, 0, values.length);
        }

        @Override
        public void toArray(float[] target) {
          throw new UnsupportedOperationException();
        }

        @Override
        public void toArray(byte[] target) {
          throw new UnsupportedOperationException();
        }
      };
    }

    @Override
    public void toArray(int[][] target) {
      for (int i = 0; i < rows.length; i++) {
        System.arraycopy(rows[i], 0, target[i], 0, rows[i].length);
      }
    }

    @Override
    public void toArray(float[][] target) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void toArray(byte[][] target) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void toHost(CuVSHostMatrix target) {
      throw new UnsupportedOperationException();
    }

    @Override
    public CuVSHostMatrix toHost() {
      throw new UnsupportedOperationException();
    }

    @Override
    public void toDevice(CuVSDeviceMatrix target, CuVSResources resources) {
      throw new UnsupportedOperationException();
    }

    @Override
    public CuVSDeviceMatrix toDevice(CuVSResources resources) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void close() {}
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CuVSDeviceMatrix;
import com.nvidia.cuvs.CuVSHostMatrix;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.RowView;
import com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.QuantizationType;
import java.io.IOException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.hnsw.HnswGraph.NodesIterator;
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
  public void testRemapPreservesSentinelAndMapsAbsoluteOrdinals() throws IOException {
    int[] destination = new int[3];

    AcceleratedHNSWUtils.remapSubsetAdjacencyRow(
        0, new IntRow(-1, 2, 0), 3, new int[] {2, 7, 11}, destination);

    assertArrayEquals(new int[] {-1, 11, 2}, destination);
  }

  @Test
  public void testRemapRejectsPositiveOrdinalOutsideSubset() {
    IOException thrown =
        assertThrows(
            IOException.class,
            () ->
                AcceleratedHNSWUtils.remapSubsetAdjacencyRow(
                    4, new IntRow(0, 3), 2, new int[] {2, 7, 11}, new int[2]));

    assertTrue(thrown.getMessage().contains("row 4, column 1"));
    assertTrue(thrown.getMessage().contains("outside [0, 3)"));
  }

  @Test
  public void testGraphMaterializationSkipsNativeSentinels() {
    IntMatrix adjacency = new IntMatrix(new int[][] {{-1, 1, 2}, {0, -1, 2}, {0, 1, -1}});
    List<int[]> layerNodes = new ArrayList<>();
    layerNodes.add(null);

    GPUBuiltHnswGraph graph = new GPUBuiltHnswGraph(3, 2, layerNodes, List.of(adjacency));

    assertArrayEquals(
        new int[] {1, 2},
        Arrays.copyOf(graph.getNeighbors(0, 0).nodes(), graph.getNeighbors(0, 0).size()));
    assertEquals(2, graph.getNeighbors(0, 0).size());
  }

  @Test
  public void testGraphMaterializationRejectsPositiveOrdinalOutsideGraph() {
    IntMatrix adjacency = new IntMatrix(new int[][] {{1}, {2}});
    List<int[]> layerNodes = new ArrayList<>();
    layerNodes.add(null);

    IllegalArgumentException thrown =
        assertThrows(
            IllegalArgumentException.class,
            () -> new GPUBuiltHnswGraph(2, 2, layerNodes, List.of(adjacency)));

    assertTrue(thrown.getMessage().contains("ordinal 2 outside graph size 2"));
  }

  @Test
  public void testUpperLayerNeighborLookupUsesSortedNodeOrdinals() {
    IntMatrix baseAdjacency = new IntMatrix(new int[][] {{-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}});
    IntMatrix upperAdjacency = new IntMatrix(new int[][] {{3}, {6}, {1}});
    int[] upperNodes = new int[] {1, 3, 6};
    GPUBuiltHnswGraph graph =
        new GPUBuiltHnswGraph(
            7, 2, Arrays.asList((int[]) null, upperNodes), List.of(baseAdjacency, upperAdjacency));

    Arrays.fill(upperNodes, 0);
    assertArrayEquals(new int[] {1, 3, 6}, NodesIterator.getSortedNodes(graph.getNodesOnLevel(1)));
    assertArrayEquals(new int[] {3}, neighbors(graph, 1, 1));
    assertArrayEquals(new int[] {6}, neighbors(graph, 1, 3));
    assertArrayEquals(new int[] {1}, neighbors(graph, 1, 6));
    assertNull(graph.getNeighbors(1, 0));
    assertNull(graph.getNeighbors(1, 4));
    assertNull(graph.getNeighbors(1, 7));
  }

  @Test
  public void testUpperLayerNeighborLookupPreservesUnsortedPublicInput() {
    IntMatrix baseAdjacency = new IntMatrix(new int[][] {{-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}});
    IntMatrix upperAdjacency = new IntMatrix(new int[][] {{1}, {3}, {6}});
    GPUBuiltHnswGraph graph =
        new GPUBuiltHnswGraph(
            7,
            2,
            Arrays.asList((int[]) null, new int[] {6, 1, 3}),
            List.of(baseAdjacency, upperAdjacency));

    assertArrayEquals(new int[] {1}, neighbors(graph, 1, 6));
    assertArrayEquals(new int[] {3}, neighbors(graph, 1, 1));
    assertArrayEquals(new int[] {6}, neighbors(graph, 1, 3));
    assertNull(graph.getNeighbors(1, 2));
  }

  @Test
  public void testUpperLayerNodeValidationRejectsDuplicatesAndOutOfRangeOrdinals() {
    IntMatrix baseAdjacency = new IntMatrix(new int[][] {{-1}, {-1}, {-1}, {-1}});
    IntMatrix threeRows = new IntMatrix(new int[][] {{1}, {2}, {3}});
    IllegalArgumentException duplicate =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                new GPUBuiltHnswGraph(
                    4,
                    2,
                    Arrays.asList((int[]) null, new int[] {1, 2, 1}),
                    List.of(baseAdjacency, threeRows)));
    assertTrue(duplicate.getMessage().contains("duplicate node ordinal 1"));

    IntMatrix twoRows = new IntMatrix(new int[][] {{1}, {2}});
    IllegalArgumentException outOfRange =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                new GPUBuiltHnswGraph(
                    4,
                    2,
                    Arrays.asList((int[]) null, new int[] {1, 4}),
                    List.of(baseAdjacency, twoRows)));
    assertTrue(outOfRange.getMessage().contains("ordinal 4 outside [0, 4)"));
  }

  private static int[] neighbors(GPUBuiltHnswGraph graph, int level, int node) {
    var neighbors = graph.getNeighbors(level, node);
    return Arrays.copyOf(neighbors.nodes(), neighbors.size());
  }

  private record IntRow(int... values) implements RowView {
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
    public void toArray(int[] array) {
      System.arraycopy(values, 0, array, 0, values.length);
    }

    @Override
    public void toArray(float[] array) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void toArray(byte[] array) {
      throw new UnsupportedOperationException();
    }
  }

  private record IntMatrix(int[][] values) implements CuVSMatrix {
    @Override
    public long size() {
      return values.length;
    }

    @Override
    public long columns() {
      return values[0].length;
    }

    @Override
    public DataType dataType() {
      return DataType.INT;
    }

    @Override
    public RowView getRow(long row) {
      return new IntRow(values[Math.toIntExact(row)]);
    }

    @Override
    public void toArray(int[][] array) {
      for (int row = 0; row < values.length; row++) {
        System.arraycopy(values[row], 0, array[row], 0, values[row].length);
      }
    }

    @Override
    public void toArray(float[][] array) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void toArray(byte[][] array) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void toHost(CuVSHostMatrix hostMatrix) {
      throw new UnsupportedOperationException();
    }

    @Override
    public CuVSHostMatrix toHost() {
      throw new UnsupportedOperationException();
    }

    @Override
    public void toDevice(CuVSDeviceMatrix deviceMatrix, CuVSResources resources) {
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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import com.nvidia.cuvs.CuVSDeviceMatrix;
import com.nvidia.cuvs.CuVSHostMatrix;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.RowView;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.NeighborArray;

/**
 * This class holds the in-memory representation of the HNSW graph
 *
 * @since 25.10
 */
public class GPUBuiltHnswGraph extends HnswGraph {

  private final int size;
  private final int dimensions;
  private final int numLevels;

  // Store layers data - each layer has its own nodes and adjacency lists
  private final List<int[]> layerNodes;
  private final List<NeighborArray[]> layerNeighbors;

  // Layer 0 is special - it contains all nodes
  private final NeighborArray[] layer0Neighbors;

  private record MaterializedGraph(
      int numLevels,
      List<int[]> layerNodes,
      NeighborArray[] layer0Neighbors,
      List<NeighborArray[]> layerNeighbors) {}

  /**
   * Multi-layer constructor that supports arbitrary number of layers.
   *
   * @param size the size of the dataset
   * @param dimensions the vector dimension
   * @param layerNodes the nodes on the layer
   * @param layerAdjacencies adjacency list
   */
  public GPUBuiltHnswGraph(
      int size, int dimensions, List<int[]> layerNodes, List<CuVSMatrix> layerAdjacencies) {
    this(size, dimensions, materializeSerial(size, layerNodes, layerAdjacencies));
  }

  private GPUBuiltHnswGraph(int size, int dimensions, MaterializedGraph graph) {
    this.size = size;
    this.dimensions = dimensions;
    this.numLevels = graph.numLevels();
    this.layerNodes = graph.layerNodes();
    this.layerNeighbors = graph.layerNeighbors();
    this.layer0Neighbors = graph.layer0Neighbors();
  }

  /** Builds a graph while bounding adjacency materialization to the requested writer threads. */
  static GPUBuiltHnswGraph create(
      int size,
      int dimensions,
      List<int[]> layerNodes,
      List<CuVSMatrix> layerAdjacencies,
      int requestedWorkers)
      throws IOException {
    return create(
        size,
        dimensions,
        layerNodes,
        layerAdjacencies,
        requestedWorkers,
        Runtime.getRuntime().availableProcessors());
  }

  static GPUBuiltHnswGraph create(
      int size,
      int dimensions,
      List<int[]> layerNodes,
      List<CuVSMatrix> layerAdjacencies,
      int requestedWorkers,
      int availableProcessors)
      throws IOException {
    if (requestedWorkers <= 1 || size < PARALLEL_MIN_NODES) {
      return new GPUBuiltHnswGraph(size, dimensions, layerNodes, layerAdjacencies);
    }
    try (BoundedParallelExecutor executor =
        BoundedParallelExecutor.create(requestedWorkers, size, availableProcessors)) {
      if (!executor.isParallel()) {
        return new GPUBuiltHnswGraph(size, dimensions, layerNodes, layerAdjacencies);
      }
      return new GPUBuiltHnswGraph(
          size, dimensions, materializeParallel(size, layerNodes, layerAdjacencies, executor));
    }
  }

  /** Node count below which parallel materialization costs more than it saves. */
  static final int PARALLEL_MIN_NODES = 1 << 16;

  /** Maximum temporary host copy used to make a device adjacency safe for concurrent reads. */
  static final long MAX_PARALLEL_GRAPH_COPY_BYTES = 4L << 30;

  private static MaterializedGraph materializeSerial(
      int size, List<int[]> layerNodes, List<CuVSMatrix> layerAdjacencies) {
    List<int[]> upperLayerNodes = new ArrayList<>();
    List<NeighborArray[]> upperLayerNeighbors = new ArrayList<>();
    NeighborArray[] baseLayerNeighbors = fillNeighborArraySerial(layerAdjacencies.get(0), size);

    for (int level = 1; level < layerAdjacencies.size(); level++) {
      int[] nodes = layerNodes.get(level);
      upperLayerNodes.add(nodes);
      upperLayerNeighbors.add(fillNeighborArraySerial(layerAdjacencies.get(level), nodes.length));
    }
    return new MaterializedGraph(
        layerAdjacencies.size(), upperLayerNodes, baseLayerNeighbors, upperLayerNeighbors);
  }

  private static MaterializedGraph materializeParallel(
      int size,
      List<int[]> layerNodes,
      List<CuVSMatrix> layerAdjacencies,
      BoundedParallelExecutor executor)
      throws IOException {
    List<int[]> upperLayerNodes = new ArrayList<>();
    List<NeighborArray[]> upperLayerNeighbors = new ArrayList<>();
    NeighborArray[] baseLayerNeighbors = fillNeighborArray(layerAdjacencies.get(0), size, executor);

    for (int level = 1; level < layerAdjacencies.size(); level++) {
      int[] nodes = layerNodes.get(level);
      upperLayerNodes.add(nodes);
      upperLayerNeighbors.add(
          fillNeighborArray(layerAdjacencies.get(level), nodes.length, executor));
    }
    return new MaterializedGraph(
        layerAdjacencies.size(), upperLayerNodes, baseLayerNeighbors, upperLayerNeighbors);
  }

  private static NeighborArray[] fillNeighborArray(
      CuVSMatrix adjacency, int size, BoundedParallelExecutor executor) throws IOException {
    if (size < PARALLEL_MIN_NODES
        || !executor.isParallel()
        || (adjacency instanceof CuVSDeviceMatrix
            && !fitsParallelGraphCopyBudget(adjacency.size(), adjacency.columns()))) {
      return fillNeighborArraySerial(adjacency, size);
    }

    if (adjacency instanceof CuVSDeviceMatrix deviceAdjacency) {
      try (CuVSHostMatrix hostCopy = copyToHost(deviceAdjacency)) {
        return fillNeighborArrayParallel(hostCopy, size, executor);
      }
    }
    return fillNeighborArrayParallel(adjacency, size, executor);
  }

  /** Returns whether an INT32 graph can be copied without exceeding the host-memory budget. */
  static boolean fitsParallelGraphCopyBudget(long rows, long columns) {
    if (rows < 0 || columns < 0) {
      return false;
    }
    if (rows == 0 || columns == 0) {
      return true;
    }
    return rows <= MAX_PARALLEL_GRAPH_COPY_BYTES / Integer.BYTES / columns;
  }

  private static CuVSHostMatrix copyToHost(CuVSDeviceMatrix source) {
    return copyToHost(
        source,
        () -> CuVSMatrix.hostBuilder(source.size(), source.columns(), source.dataType()).build());
  }

  static CuVSHostMatrix copyToHost(
      CuVSDeviceMatrix source, Supplier<CuVSHostMatrix> hostCopyFactory) {
    CuVSHostMatrix hostCopy = hostCopyFactory.get();
    try {
      source.toHost(hostCopy);
      return hostCopy;
    } catch (RuntimeException | Error failure) {
      try {
        hostCopy.close();
      } catch (RuntimeException | Error closeFailure) {
        if (failure != closeFailure) {
          failure.addSuppressed(closeFailure);
        }
      }
      throw failure;
    }
  }

  private static NeighborArray[] fillNeighborArraySerial(CuVSMatrix adjacency, int size) {
    NeighborArray[] neighbors = new NeighborArray[size];
    for (int i = 0; i < size; i++) {
      neighbors[i] = materializeRow(adjacency.getRow(i));
    }
    return neighbors;
  }

  private static NeighborArray[] fillNeighborArrayParallel(
      CuVSMatrix adjacency, int size, BoundedParallelExecutor executor) throws IOException {
    NeighborArray[] neighbors = new NeighborArray[size];
    executor.invokeRanges(
        size,
        (taskIndex, start, end) -> {
          for (int row = start; row < end; row++) {
            if ((row & 0x3ff) == 0 && Thread.currentThread().isInterrupted()) {
              throw new InterruptedException("graph materialization interrupted");
            }
            neighbors[row] = materializeRow(adjacency.getRow(row));
          }
        });
    return neighbors;
  }

  private static NeighborArray materializeRow(RowView row) {
    if (row == null || row.size() == 0) {
      return new NeighborArray(0, true);
    }
    NeighborArray neighbors = new NeighborArray(Math.toIntExact(row.size()), true);
    for (int index = 0; index < row.size(); index++) {
      neighbors.addInOrder(row.getAsInt(index), 1.0f - (index * 0.001f));
    }
    return neighbors;
  }

  /**
   * Get all nodes on a given level as node 0th ordinals.
   */
  public NodesIterator getNodesOnLevel(int level) {
    if (level == 0) {
      return new Level0NodesIterator(size);
    } else if (level > 0 && level < numLevels) {
      int[] nodes = layerNodes.get(level - 1);
      return new HigherLevelNodesIterator(nodes);
    } else {
      return new Level0NodesIterator(0);
    }
  }

  /**
   * Get the neighbors for the node and the level it resides.
   *
   * @param level the level
   * @param node the node
   * @return an instance of NeighborArray
   */
  public NeighborArray getNeighbors(int level, int node) {
    if (level == 0 && node < size) {
      return layer0Neighbors[node];
    } else if (level > 0 && level < numLevels) {
      int[] nodes = layerNodes.get(level - 1);
      NeighborArray[] neighbors = layerNeighbors.get(level - 1);

      // Find the index of this node in the layer
      for (int i = 0; i < nodes.length; i++) {
        if (nodes[i] == node) {
          return neighbors[i];
        }
      }
    }
    return null;
  }

  // Implementation of abstract methods from HnswGraph
  private int currentNode = -1;
  private int currentLevel = -1;
  private int neighborIndex = -1;

  /**
   * Move the pointer to exactly the given level's target.
   */
  @Override
  public void seek(int level, int target) {
    currentLevel = level;
    currentNode = target;
    neighborIndex = -1;
  }

  /**
   * Iterates over the neighbor list.
   */
  @Override
  public int nextNeighbor() {
    if (currentLevel == 0
        && currentNode >= 0
        && currentNode < size
        && layer0Neighbors[currentNode] != null) {
      neighborIndex++;
      if (neighborIndex < layer0Neighbors[currentNode].size()) {
        int neighborNode = layer0Neighbors[currentNode].nodes()[neighborIndex];
        if (neighborNode >= 0 && neighborNode < size) {
          return neighborNode;
        } else {
          return nextNeighbor(); // Skip invalid neighbor
        }
      }
    } else if (currentLevel > 0 && currentLevel < numLevels) {
      // Handle higher layers
      NeighborArray neighbors = getNeighbors(currentLevel, currentNode);
      if (neighbors != null) {
        neighborIndex++;
        if (neighborIndex < neighbors.size()) {
          return neighbors.nodes()[neighborIndex];
        }
      }
    }
    return NO_MORE_DOCS;
  }

  /**
   * Returns graph's entry point on the top level.
   */
  @Override
  public int entryNode() {
    // Entry node should be from the highest layer
    if (numLevels > 1) {
      int topLevel = numLevels - 1;
      int[] topLayerNodes = layerNodes.get(topLevel - 1);
      if (topLayerNodes != null && topLayerNodes.length > 0) {
        // Use random node from top layer with fixed seed for reproducibility
        java.util.Random random = new java.util.Random(44);
        int randomIndex = random.nextInt(topLayerNodes.length);
        return topLayerNodes[randomIndex];
      }
    }
    return 0; // Default to node 0 for single-layer graphs
  }

  /**
   * returns M, the maximum number of connections for a node.
   */
  @Override
  public int maxConn() {
    // Return the maximum degree across all nodes in layer 0
    int max = 0;
    for (NeighborArray neighbor : layer0Neighbors) {
      if (neighbor != null) {
        max = Math.max(max, neighbor.size());
      }
    }
    return max;
  }

  /**
   * Returns the neighbor count.
   */
  @Override
  public int neighborCount() {
    if (currentLevel == 0
        && currentNode >= 0
        && currentNode < size
        && layer0Neighbors[currentNode] != null) {
      return layer0Neighbors[currentNode].size();
    } else if (currentLevel > 0 && currentLevel < numLevels) {
      NeighborArray neighbors = getNeighbors(currentLevel, currentNode);
      return neighbors != null ? neighbors.size() : 0;
    }
    return 0;
  }

  // NodesIterator for level 0
  private static class Level0NodesIterator extends NodesIterator {
    private int current = -1;

    Level0NodesIterator(int size) {
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
      int numToCopy = Math.min(dest.length, size - (current + 1));
      for (int i = 0; i < numToCopy; i++) {
        dest[i] = ++current;
      }
      return numToCopy;
    }
  }

  // NodesIterator for higher layers
  private static class HigherLevelNodesIterator extends NodesIterator {
    private final int[] nodeIds;
    private int current = -1;

    HigherLevelNodesIterator(int[] nodeIds) {
      super(nodeIds.length);
      this.nodeIds = nodeIds;
    }

    @Override
    public boolean hasNext() {
      return current + 1 < nodeIds.length;
    }

    @Override
    public int nextInt() {
      return nodeIds[++current];
    }

    @Override
    public int consume(int[] dest) {
      int numToCopy = Math.min(dest.length, nodeIds.length - (current + 1));
      for (int i = 0; i < numToCopy; i++) {
        dest[i] = nodeIds[++current];
      }
      return numToCopy;
    }
  }

  /**
   * Returns the number of nodes in the graph.
   */
  public int size() {
    return size;
  }

  /**
   * Returns the number of levels in the HNSW graph.
   *
   * @return the number of levels
   */
  public int numLevels() {
    return numLevels;
  }

  /**
   * Gets the vector dimension.
   *
   * @return the vector dimension
   */
  public int dimensions() {
    return dimensions;
  }
}

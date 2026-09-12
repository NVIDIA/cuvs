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
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.apache.lucene.search.TaskExecutor;
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

  /**
   * Multi-layer constructor that supports arbitrary number of layers.
   *
   * @param size the size of the dataset
   * @param dimensions the vector dimension
   * @param layerNodes the nodes on the layer
   * @param layerAdjacencies adjacency list
   * @param numThreads threads to use for materializing the adjacency (1 = serial)
   */
  public GPUBuiltHnswGraph(
      int size,
      int dimensions,
      List<int[]> layerNodes,
      List<CuVSMatrix> layerAdjacencies,
      int numThreads)
      throws IOException {

    this.size = size;
    this.dimensions = dimensions;
    this.numLevels = layerAdjacencies.size();
    this.layerNodes = new ArrayList<>();
    this.layerNeighbors = new ArrayList<>();

    // Process Layer 0 (base layer with all nodes)
    CuVSMatrix layer0Adjacency = layerAdjacencies.get(0);
    this.layer0Neighbors = fillNeighborArray(layer0Adjacency, size, numThreads);

    // Process higher layers (1 to numLevels-1)
    for (int level = 1; level < numLevels; level++) {
      int[] nodes = layerNodes.get(level);
      CuVSMatrix adjacency = layerAdjacencies.get(level);
      this.layerNodes.add(nodes);
      this.layerNeighbors.add(fillNeighborArray(adjacency, nodes.length, numThreads));
    }
  }

  /** Node count below which parallel materialization is not worth the thread overhead. */
  private static final int PARALLEL_MIN_NODES = 1 << 16;

  /**
   * Materializes the adjacency matrix into on-heap {@link NeighborArray}s, one per node.
   *
   * <p>The serial path reads the adjacency directly (a device matrix's {@code getRow} is safe
   * single-threaded). The parallel path cannot: the CAGRA layer-0 adjacency is a device matrix whose
   * {@code getRow} uses a shared, stateful buffered reader that is not safe for concurrent access, so
   * it is pulled to host once (a single bulk device-&gt;host copy) before materializing disjoint node
   * ranges concurrently. Host matrices (the upper layers, built via {@link CuVSMatrix#ofArray}) are
   * read directly in both paths.
   *
   * @param adjacency instance of adjacency CuVSMatrix
   * @param size the number of nodes
   * @param numThreads threads to use (1, or fewer than {@value #PARALLEL_MIN_NODES} nodes = serial)
   * @return the NeighborArray
   */
  private static NeighborArray[] fillNeighborArray(CuVSMatrix adjacency, int size, int numThreads)
      throws IOException {
    NeighborArray[] neighbors = new NeighborArray[size];
    if (numThreads <= 1 || size < PARALLEL_MIN_NODES) {
      fillNeighborRange(adjacency, neighbors, 0, size);
      return neighbors;
    }
    CuVSMatrix source = adjacency;
    CuVSHostMatrix hostCopy = null;
    if (adjacency instanceof CuVSDeviceMatrix deviceAdjacency) {
      hostCopy = deviceAdjacency.toHost();
      source = hostCopy;
    }
    try {
      fillNeighborArrayParallel(source, neighbors, size, numThreads);
      return neighbors;
    } finally {
      if (hostCopy != null) {
        hostCopy.close();
      }
    }
  }

  /**
   * Materializes disjoint node ranges concurrently. Each thread writes its own slots of {@code
   * neighbors} and its own {@link NeighborArray} instances, so no synchronization is needed; {@code
   * source} must be a host matrix (stateless {@code getRow}).
   */
  private static void fillNeighborArrayParallel(
      CuVSMatrix source, NeighborArray[] neighbors, int size, int numThreads) throws IOException {
    ExecutorService pool = Executors.newFixedThreadPool(Math.max(1, numThreads - 1));
    try {
      int perThread = (size + numThreads - 1) / numThreads;
      List<Callable<Void>> tasks = new ArrayList<>(numThreads);
      for (int t = 0; t < numThreads; t++) {
        final int start = t * perThread;
        final int end = Math.min(start + perThread, size);
        if (start >= end) {
          break;
        }
        tasks.add(
            () -> {
              fillNeighborRange(source, neighbors, start, end);
              return null;
            });
      }
      new TaskExecutor(pool).invokeAll(tasks);
    } finally {
      pool.shutdown();
    }
  }

  /** Fills {@code neighbors[start, end)} from the adjacency rows. */
  private static void fillNeighborRange(
      CuVSMatrix source, NeighborArray[] neighbors, int start, int end) {
    for (int i = start; i < end; i++) {
      RowView rv = source.getRow(i);
      if (rv != null && rv.size() > 0) {
        NeighborArray na = new NeighborArray((int) rv.size(), true);
        for (int j = 0; j < rv.size(); j++) {
          na.addInOrder(rv.getAsInt(j), 1.0f - (j * 0.001f));
        }
        neighbors[i] = na;
      } else {
        neighbors[i] = new NeighborArray(0, true);
      }
    }
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

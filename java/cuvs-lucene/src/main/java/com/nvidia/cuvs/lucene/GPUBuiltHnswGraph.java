/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.RowView;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
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
  private final boolean[] sortedUpperLayers;

  // Layer 0 is special - it contains all nodes
  private final NeighborArray[] layer0Neighbors;

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
    if (size <= 0) {
      throw new IllegalArgumentException("Graph size must be positive");
    }
    if (dimensions <= 0) {
      throw new IllegalArgumentException("Vector dimensions must be positive");
    }
    if (layerAdjacencies.isEmpty() || layerNodes.size() != layerAdjacencies.size()) {
      throw new IllegalArgumentException(
          "Layer node and adjacency lists must have the same non-zero size");
    }
    this.size = size;
    this.dimensions = dimensions;
    this.numLevels = layerAdjacencies.size();
    this.layerNodes = new ArrayList<>();
    this.layerNeighbors = new ArrayList<>();
    this.sortedUpperLayers = new boolean[Math.max(0, numLevels - 1)];

    // Process Layer 0 (base layer with all nodes)
    CuVSMatrix layer0Adjacency = layerAdjacencies.get(0);
    this.layer0Neighbors = fillNeighborArray(layer0Adjacency, size);

    // Process higher layers (1 to numLevels-1)
    HashSet<Integer> previousLayerNodes = null;
    for (int level = 1; level < numLevels; level++) {
      int[] suppliedNodes = layerNodes.get(level);
      if (suppliedNodes == null) {
        throw new IllegalArgumentException("Missing node ordinals for level " + level);
      }
      int[] nodes = suppliedNodes.clone();
      HashSet<Integer> currentLayerNodes = new HashSet<>(nodes.length);
      sortedUpperLayers[level - 1] =
          validateLayerNodes(nodes, level, previousLayerNodes, currentLayerNodes);
      CuVSMatrix adjacency = layerAdjacencies.get(level);
      NeighborArray[] neighbors = fillNeighborArray(adjacency, nodes.length);
      validateUpperLayerNeighbors(nodes, neighbors, currentLayerNodes, level);
      this.layerNodes.add(nodes);
      this.layerNeighbors.add(neighbors);
      previousLayerNodes = currentLayerNodes;
    }
  }

  /**
   * Fills the neighbor array using the adjacency matrix.
   *
   * @param adjacency instance of adjacency CuVSMatrix
   * @param size the number of nodes
   * @return the NeighborArray
   */
  private NeighborArray[] fillNeighborArray(CuVSMatrix adjacency, int size) {
    if (adjacency.dataType() != CuVSMatrix.DataType.INT
        && adjacency.dataType() != CuVSMatrix.DataType.UINT) {
      throw new IllegalArgumentException(
          "Expected INT or UINT adjacency data, but received " + adjacency.dataType());
    }
    if (adjacency.size() != size) {
      throw new IllegalArgumentException(
          "Expected " + size + " adjacency rows, but received " + adjacency.size());
    }
    int degree = Math.toIntExact(adjacency.columns());
    if (degree <= 0) {
      throw new IllegalArgumentException("Adjacency matrices must have a positive degree");
    }
    NeighborArray[] neighbors = new NeighborArray[size];
    for (int i = 0; i < size; i++) {
      RowView rv = adjacency.getRow(i);
      if (rv == null || rv.size() != degree) {
        throw new IllegalArgumentException(
            "Expected "
                + degree
                + " neighbors for adjacency row "
                + i
                + ", but received "
                + (rv == null ? "null" : rv.size()));
      }
      neighbors[i] = new NeighborArray(degree, true);
      for (int j = 0; j < degree; j++) {
        int neighbor = rv.getAsInt(j);
        if (neighbor < 0) {
          continue;
        }
        if (neighbor >= this.size) {
          throw new IllegalArgumentException(
              "Adjacency row "
                  + i
                  + " contains ordinal "
                  + neighbor
                  + " outside graph size "
                  + this.size);
        }
        neighbors[i].addInOrder(neighbor, 1.0f - (j * 0.001f));
      }
    }
    return neighbors;
  }

  private boolean validateLayerNodes(
      int[] nodes,
      int level,
      HashSet<Integer> previousLayerNodes,
      HashSet<Integer> currentLayerNodes) {
    if (nodes.length == 0) {
      throw new IllegalArgumentException("Level " + level + " must contain at least one node");
    }
    boolean sorted = true;
    int previous = -1;
    for (int node : nodes) {
      if (node < 0 || node >= size) {
        throw new IllegalArgumentException(
            "Level " + level + " contains node ordinal " + node + " outside [0, " + size + ")");
      }
      if (previousLayerNodes != null && previousLayerNodes.contains(node) == false) {
        throw new IllegalArgumentException(
            "Level "
                + level
                + " contains node ordinal "
                + node
                + " that is absent from level "
                + (level - 1));
      }
      if (currentLayerNodes.add(node) == false) {
        throw new IllegalArgumentException(
            "Level " + level + " contains duplicate node ordinal " + node);
      }
      sorted &= node > previous;
      previous = node;
    }
    return sorted;
  }

  private static void validateUpperLayerNeighbors(
      int[] nodes, NeighborArray[] neighbors, HashSet<Integer> currentLayerNodes, int level) {
    for (int row = 0; row < neighbors.length; row++) {
      NeighborArray rowNeighbors = neighbors[row];
      for (int column = 0; column < rowNeighbors.size(); column++) {
        int neighbor = rowNeighbors.nodes()[column];
        if (currentLayerNodes.contains(neighbor) == false) {
          throw new IllegalArgumentException(
              "Level "
                  + level
                  + " adjacency for node "
                  + nodes[row]
                  + " contains neighbor ordinal "
                  + neighbor
                  + " that is absent from level "
                  + level);
        }
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
      if (sortedUpperLayers[level - 1]) {
        int index = Arrays.binarySearch(nodes, node);
        return index >= 0 ? neighbors[index] : null;
      }
      for (int index = 0; index < nodes.length; index++) {
        if (nodes[index] == node) {
          return neighbors[index];
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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.getCuVSResourcesInstance;
import static com.nvidia.cuvs.lucene.Utils.createByteMatrixFromArray;

import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.RowView;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.SortedSet;
import java.util.TreeSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.HnswGraph.NodesIterator;
import org.apache.lucene.util.hnsw.NeighborArray;
import org.apache.lucene.util.packed.DirectMonotonicWriter;

public class AcceleratedHNSWUtils {

  public enum QuantizationType {
    BINARY,
    SCALAR
  }

  private static final LuceneProvider LUCENE_PROVIDER;
  private static final List<VectorSimilarityFunction> VECTOR_SIMILARITY_FUNCTIONS;

  static {
    try {
      LUCENE_PROVIDER = LuceneProvider.getInstance("99");
      VECTOR_SIMILARITY_FUNCTIONS = LUCENE_PROVIDER.getSimilarityFunctions();
    } catch (Exception e) {
      throw new ExceptionInInitializerError(e.getMessage());
    }
  }

  /**
   * Creates a dummy HNSW graph for a single vector.
   * The graph will have 1 level with 1 node and no neighbors.
   */
  public static GPUBuiltHnswGraph createSingleVectorHnswGraph(int size, int dimensions)
      throws Throwable {
    // Create adjacency list for single node with no neighbors
    int[][] singleNodeAdjacency = new int[][] {{-1}}; // -1 indicates no neighbors

    // Create CuVSMatrix from the adjacency list
    CuVSMatrix adjacencyMatrix = CuVSMatrix.ofArray(singleNodeAdjacency);

    // Create layer data for single-level graph
    List<int[]> layerNodes = new ArrayList<>();
    List<CuVSMatrix> layerAdjacencies = new ArrayList<>();

    // Layer 0: contains all nodes (just the single node)
    layerNodes.add(null); // Layer 0 contains all nodes, so we don't need to store node list
    layerAdjacencies.add(adjacencyMatrix);

    // Create the single-layer graph
    return new GPUBuiltHnswGraph(size, dimensions, layerNodes, layerAdjacencies);
  }

  /**
   * Creates a multi-layer HNSW graph with dynamic number of layers.
   * M = cagraGraphDegree/2
   * Each layer contains 1/M nodes from the previous layer
   * Creates layers until the highest layer has ≤ M nodes
   */
  public static GPUBuiltHnswGraph createMultiLayerHnswGraph(
      FieldInfo fieldInfo,
      int size,
      int dimensions,
      CuVSMatrix adjacencyListMatrix,
      List<byte[]> vectors,
      int hnswLayers,
      int graphDegree,
      CagraIndexParams params,
      QuantizationType quantization)
      throws Throwable {

    // Calculate M as cagraGraphDegree/2
    int M = graphDegree / 2;

    // Store all layers data
    List<int[]> layerNodes = new ArrayList<>();
    List<CuVSMatrix> layerAdjacencies = new ArrayList<>();

    // Layer 0: Use full CAGRA adjacency list
    layerNodes.add(null); // Layer 0 contains all nodes, so we don't need to store node list
    layerAdjacencies.add(adjacencyListMatrix);

    int currentLayerSize = size;
    int layerIndex = 1;
    Random random = new Random();

    while (layerIndex < hnswLayers && currentLayerSize > 1) {
      // Calculate size for next layer (1/M of current layer)
      int nextLayerSize = Math.max(2, currentLayerSize / M);
      // Select nodes for this layer
      SortedSet<Integer> selectedNodesSet = new TreeSet<>();

      if (layerIndex == 1) {
        // Select from all nodes (Layer 0)
        while (selectedNodesSet.size() < nextLayerSize) {
          selectedNodesSet.add(random.nextInt(size));
        }
      } else {
        // Select from previous layer nodes
        int[] prevLayerNodes = layerNodes.get(layerNodes.size() - 1);
        while (selectedNodesSet.size() < nextLayerSize) {
          int idx = random.nextInt(prevLayerNodes.length);
          selectedNodesSet.add(prevLayerNodes[idx]);
        }
      }

      // Convert to sorted array
      int[] selectedNodes =
          selectedNodesSet.stream().mapToInt(Integer::intValue).sorted().toArray();

      layerNodes.add(selectedNodes);

      // Extract vectors for selected nodes
      int bytesPerVector = (dimensions + 7) / 8;
      byte[][] selectedVectors = new byte[nextLayerSize][];
      for (int i = 0; i < nextLayerSize; i++) {
        selectedVectors[i] = vectors.get(selectedNodes[i]);
      }

      // Build CAGRA graph for this layer
      layerAdjacencies.add(
          buildCagraGraphForSubset(
              selectedVectors, selectedNodes, bytesPerVector, params, dimensions, quantization));

      // Update for next iteration
      currentLayerSize = nextLayerSize;
      layerIndex++;

      // Use different seed for each layer
      random = new Random(new Random().nextLong());
    }

    // Create the multi-layer graph with all layers
    return new GPUBuiltHnswGraph(size, dimensions, layerNodes, layerAdjacencies);
  }

  /**
   * Builds a CAGRA graph for a subset of binary quantized vectors
   */
  private static CuVSMatrix buildCagraGraphForSubset(
      byte[][] vectors,
      int[] selectedNodes,
      int bytesPerVector,
      CagraIndexParams params,
      int dimensions,
      QuantizationType quantization)
      throws Throwable {

    CuVSMatrix subsetDataset;

    if (quantization == QuantizationType.BINARY) {
      subsetDataset =
          createByteMatrixFromArray(vectors, bytesPerVector, getCuVSResourcesInstance());
    } else {
      subsetDataset = createByteMatrixFromArray(vectors, dimensions, getCuVSResourcesInstance());
    }

    // Build CAGRA index for the subset
    CagraIndex subsetIndex =
        CagraIndex.newBuilder(getCuVSResourcesInstance())
            .withDataset(subsetDataset)
            .withIndexParams(params)
            .build();

    // Get adjacency list from subset CAGRA index
    CuVSMatrix cagraGraph = subsetIndex.getGraph();

    long numNodes = cagraGraph.size();
    long degree = cagraGraph.columns();

    // Create a re-mapped adjacency list
    int[][] remappedAdjacency = new int[(int) numNodes][(int) degree];

    for (int i = 0; i < numNodes; i++) {
      RowView rv = cagraGraph.getRow(i);
      for (int j = 0; j < degree && j < rv.size(); j++) {
        int subsetIndex1 = rv.getAsInt(j);
        // Map subset index to original node ID
        if (subsetIndex1 >= 0 && subsetIndex1 < selectedNodes.length) {
          remappedAdjacency[i][j] = selectedNodes[subsetIndex1];
        } else {
          // Invalid index, use self-reference
          remappedAdjacency[i][j] = selectedNodes[i];
        }
      }
    }

    subsetIndex.close();
    return CuVSMatrix.ofArray(remappedAdjacency);
  }

  /**
   * Returns a 2D array of offsets (information written while writing the meta info)
   *
   * @param graph instance of GPUBuiltHnswGraph
   * @param vectorIndex instance of IndexOutput
   * @return a 2D array of offsets
   * @throws IOException I/O Exceptions
   */
  public static int[][] writeGraph(GPUBuiltHnswGraph graph, IndexOutput vectorIndex)
      throws IOException {
    // write vectors' neighbors on each level into the vectorIndex file
    int countOnLevel0 = graph.size();
    int[][] offsets = new int[graph.numLevels()][];
    int[] scratch = new int[graph.maxConn() * 2];
    for (int level = 0; level < graph.numLevels(); level++) {
      int[] sortedNodes = NodesIterator.getSortedNodes(graph.getNodesOnLevel(level));
      offsets[level] = new int[sortedNodes.length];
      int nodeOffsetId = 0;

      for (int node : sortedNodes) {
        // Get node neighbors
        NeighborArray neighbors = graph.getNeighbors(level, node);
        // Get the size of the neighbor array
        int size = neighbors.size();
        // Write size in VInt as the neighbors list is typically small
        long offsetStart = vectorIndex.getFilePointer();
        // Get neighbors
        int[] nnodes = neighbors.nodes();
        // Sort them
        Arrays.sort(nnodes, 0, size);
        // Now that we have sorted, do delta encoding to minimize the required bits to store the
        // information
        int actualSize = 0;
        if (size > 0) {
          scratch[0] = nnodes[0];
          actualSize = 1;
        }
        // De-duplication
        for (int i = 1; i < size; i++) {
          assert nnodes[i] < countOnLevel0 : "node too large: " + nnodes[i] + ">=" + countOnLevel0;
          // Sorting step helps here
          if (nnodes[i - 1] == nnodes[i]) {
            continue;
          }
          scratch[actualSize++] = nnodes[i] - nnodes[i - 1];
        }
        // Write the size after duplicates are removed
        vectorIndex.writeVInt(actualSize);
        // Write de-duplicated neighbors
        for (int i = 0; i < actualSize; i++) {
          vectorIndex.writeVInt(scratch[i]);
        }
        offsets[level][nodeOffsetId++] =
            Math.toIntExact(vectorIndex.getFilePointer() - offsetStart);
      }
    }
    // Return offsets (information written while writing the meta info)
    return offsets;
  }

  /**
   * Writes the meta information for the index.
   *
   * @param vectorIndex instance of IndexOutput
   * @param meta instance of IndexOutput
   * @param field instance of FieldInfo
   * @param vectorIndexOffset vector index offset
   * @param vectorIndexLength vector index length
   * @param count the count of vectors
   * @param graph instance of HnswGraph
   * @param graphLevelNodeOffsets graph level node offsets
   * @throws IOException I/O Exceptions
   */
  public static void writeMeta(
      IndexOutput vectorIndex,
      IndexOutput meta,
      FieldInfo field,
      long vectorIndexOffset,
      long vectorIndexLength,
      int count,
      HnswGraph graph,
      int[][] graphLevelNodeOffsets,
      int graphDegree)
      throws IOException {

    meta.writeInt(field.number);
    meta.writeInt(field.getVectorEncoding().ordinal());
    meta.writeInt(distFuncToOrd(field.getVectorSimilarityFunction()));
    meta.writeVLong(vectorIndexOffset);
    meta.writeVLong(vectorIndexLength);
    meta.writeVInt(field.getVectorDimension());
    meta.writeInt(count);
    meta.writeVInt(graphDegree / 2); // M = cagraGraphDegree/2

    // write graph nodes on each level
    if (graph == null) {
      meta.writeVInt(0);
    } else {
      meta.writeVInt(graph.numLevels());
      long valueCount = 0;
      for (int level = 0; level < graph.numLevels(); level++) {
        NodesIterator nodesOnLevel = graph.getNodesOnLevel(level);
        valueCount += nodesOnLevel.size();
        if (level > 0) {
          int[] nol = new int[nodesOnLevel.size()];
          int numberConsumed = nodesOnLevel.consume(nol);
          Arrays.sort(nol);
          assert numberConsumed == nodesOnLevel.size();
          meta.writeVInt(nol.length); // number of nodes on a level
          for (int i = nodesOnLevel.size() - 1; i > 0; --i) {
            nol[i] -= nol[i - 1];
          }
          for (int n : nol) {
            meta.writeVInt(n);
          }
        } else {
          assert nodesOnLevel.size() == count : "Level 0 expects to have all nodes";
        }
      }

      long start = vectorIndex.getFilePointer();
      meta.writeLong(start);
      meta.writeVInt(16); // DIRECT_MONOTONIC_BLOCK_SHIFT);

      final DirectMonotonicWriter memoryOffsetsWriter =
          DirectMonotonicWriter.getInstance(meta, vectorIndex, valueCount, 16);
      long cumulativeOffsetSum = 0;
      for (int[] levelOffsets : graphLevelNodeOffsets) {
        for (int v : levelOffsets) {
          memoryOffsetsWriter.add(cumulativeOffsetSum);
          cumulativeOffsetSum += v;
        }
      }

      memoryOffsetsWriter.finish();
      meta.writeLong(vectorIndex.getFilePointer() - start);
    }
  }

  public static int distFuncToOrd(VectorSimilarityFunction func) {
    for (int i = 0; i < VECTOR_SIMILARITY_FUNCTIONS.size(); i++) {
      if (VECTOR_SIMILARITY_FUNCTIONS.get(i).equals(func)) {
        return (byte) i;
      }
    }
    throw new IllegalArgumentException("invalid distance function: " + func);
  }

  /**
   * A utility method to print info/debugging messages using InfoStream.
   *
   * @param msg the debugging message to print
   */
  public static void printInfoStream(InfoStream infoStream, String component, String msg) {
    if (infoStream.isEnabled(component)) {
      infoStream.message(component, msg);
    }
  }

  /**
   * Writes an empty meta information for the field.
   *
   * @param fieldInfo instance of FieldInfo
   * @throws IOException I/O Exceptions
   */
  public static void writeEmpty(FieldInfo fieldInfo, IndexOutput op) throws IOException {
    writeMeta(null, op, fieldInfo, 0, 0, 0, null, null, 0);
  }

  /**
   * Builds an instance of CagraIndexParams.
   *
   * @return instance of CagraIndexParams
   */
  public static CagraIndexParams cagraIndexParams(
      int cuvsWriterThreads, int intGraphDegree, int graphDegree) {
    return new CagraIndexParams.Builder()
        .withNumWriterThreads(cuvsWriterThreads)
        .withIntermediateGraphDegree(intGraphDegree)
        .withGraphDegree(graphDegree)
        .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
        .build();
  }

  /**
   * Quantizes FLOAT32 vectors to binary (1 bit per dimension, packed into bytes).
   * Binary quantization: each dimension is compared to a centroid (mean of all values for that dimension).
   * If value > centroid, bit = 1, else bit = 0.
   * Bits are packed: 8 dimensions per byte.
   *
   * @param floatVectors A list of float vectors
   * @return A list of byte binary representation for the input vectors
   */
  public static List<byte[]> quantizeFloatVectorsToBinary(List<float[]> floatVectors) {
    if (floatVectors.isEmpty()) {
      return new ArrayList<>();
    }

    int dimensions = floatVectors.get(0).length;
    int numVectors = floatVectors.size();
    int bytesPerVector = (dimensions + 7) / 8;

    float[] centroids = new float[dimensions];
    for (float[] vector : floatVectors) {
      for (int d = 0; d < dimensions; d++) {
        centroids[d] += vector[d];
      }
    }
    for (int d = 0; d < dimensions; d++) {
      centroids[d] /= numVectors;
    }

    List<byte[]> quantizedVectors = new ArrayList<>(numVectors);
    for (float[] vector : floatVectors) {
      byte[] quantized = new byte[bytesPerVector];
      for (int d = 0; d < dimensions; d++) {
        boolean bit = vector[d] > centroids[d];
        int byteIndex = d / 8;
        int bitIndex = d % 8;
        if (bit) {
          quantized[byteIndex] |= (1 << bitIndex);
        }
      }
      quantizedVectors.add(quantized);
    }

    return quantizedVectors;
  }

  /**
   * Scalar quantization.
   *
   * @param floatVectors A list of float vectors
   * @return A list of byte scalar representation for the input vectors
   */
  public static List<byte[]> quantizeFloatVectorsToScalar(List<float[]> floatVectors) {
    if (floatVectors.isEmpty()) {
      return new ArrayList<>();
    }

    int dimensions = floatVectors.get(0).length;
    int numVectors = floatVectors.size();

    float[] minPerDim = new float[dimensions];
    float[] maxPerDim = new float[dimensions];
    Arrays.fill(minPerDim, Float.MAX_VALUE);
    Arrays.fill(maxPerDim, Float.MIN_VALUE);

    for (float[] vector : floatVectors) {
      for (int d = 0; d < dimensions; d++) {
        minPerDim[d] = Math.min(minPerDim[d], vector[d]);
        maxPerDim[d] = Math.max(maxPerDim[d], vector[d]);
      }
    }

    List<byte[]> quantizedVectors = new ArrayList<>(numVectors);
    for (float[] vector : floatVectors) {
      byte[] quantized = new byte[dimensions];
      for (int d = 0; d < dimensions; d++) {
        float range = maxPerDim[d] - minPerDim[d];
        if (range > 0) {
          float normalized = (vector[d] - minPerDim[d]) / range;
          int quantizedValue = Math.round(normalized * 127.0f) - 64;
          quantized[d] = (byte) Math.max(-64, Math.min(63, quantizedValue));
        } else {
          quantized[d] = 0;
        }
      }
      quantizedVectors.add(quantized);
    }

    return quantizedVectors;
  }
}

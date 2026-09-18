/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.getCuVSResourcesInstance;
import static com.nvidia.cuvs.lucene.Utils.createByteMatrixFromArray;

import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraIndexParams;
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
import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.store.DataOutput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.HnswGraph.NodesIterator;
import org.apache.lucene.util.hnsw.NeighborArray;
import org.apache.lucene.util.packed.DirectMonotonicWriter;

public class AcceleratedHNSWUtils {

  public enum QuantizationType {
    BINARY,
    SCALAR,
    NONE
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
   * M = ceil(cagraGraphDegree / 2), where cagraGraphDegree is the CAGRA adjacency list's degree
   * (its column count). Ceil is used to accommodate odd graph degrees.
   * Each layer contains 1/M nodes from the previous layer
   * Creates layers until the highest layer has ≤ M nodes
   */
  public static GPUBuiltHnswGraph createMultiLayerHnswGraph(
      FieldInfo fieldInfo,
      int size,
      int dimensions,
      CuVSMatrix adjacencyListMatrix,
      List<?> vectors,
      int hnswLayers,
      CagraIndexParams params,
      QuantizationType quantization)
      throws Throwable {
    return createMultiLayerHnswGraph(
        fieldInfo,
        size,
        dimensions,
        adjacencyListMatrix,
        vectors,
        hnswLayers,
        params,
        quantization,
        1);
  }

  static GPUBuiltHnswGraph createMultiLayerHnswGraph(
      FieldInfo fieldInfo,
      int size,
      int dimensions,
      CuVSMatrix adjacencyListMatrix,
      List<?> vectors,
      int hnswLayers,
      CagraIndexParams params,
      QuantizationType quantization,
      int requestedWorkers)
      throws Throwable {

    int M = Math.ceilDiv((int) adjacencyListMatrix.columns(), 2);

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

      if (quantization == QuantizationType.NONE) {
        // Extract vectors for selected nodes
        float[][] selectedVectors = new float[nextLayerSize][];
        for (int i = 0; i < nextLayerSize; i++) {
          selectedVectors[i] = (float[]) vectors.get(selectedNodes[i]);
        }

        // Build CAGRA graph for this layer
        layerAdjacencies.add(
            buildCagraGraphForSubset(
                selectedVectors, selectedNodes, 0, params, dimensions, quantization));

      } else {

        // Extract vectors for selected nodes
        int bytesPerVector = (dimensions + 7) / 8;
        byte[][] selectedVectors = new byte[nextLayerSize][];
        for (int i = 0; i < nextLayerSize; i++) {
          selectedVectors[i] = (byte[]) vectors.get(selectedNodes[i]);
        }

        // Build CAGRA graph for this layer
        layerAdjacencies.add(
            buildCagraGraphForSubset(
                selectedVectors, selectedNodes, bytesPerVector, params, dimensions, quantization));
      }

      // Update for next iteration
      currentLayerSize = nextLayerSize;
      layerIndex++;

      // Use different seed for each layer
      random = new Random(new Random().nextLong());
    }

    // Create the multi-layer graph with all layers
    return GPUBuiltHnswGraph.create(
        size, dimensions, layerNodes, layerAdjacencies, requestedWorkers);
  }

  /**
   * Builds a CAGRA graph for a subset of binary quantized vectors
   */
  private static CuVSMatrix buildCagraGraphForSubset(
      Object vectors,
      int[] selectedNodes,
      int bytesPerVector,
      CagraIndexParams params,
      int dimensions,
      QuantizationType quantization)
      throws Throwable {

    CuVSMatrix subsetDataset;

    if (quantization == QuantizationType.BINARY) {
      subsetDataset =
          createByteMatrixFromArray((byte[][]) vectors, bytesPerVector, getCuVSResourcesInstance());
    } else if (quantization == QuantizationType.SCALAR) {
      subsetDataset =
          createByteMatrixFromArray((byte[][]) vectors, dimensions, getCuVSResourcesInstance());
    } else {
      subsetDataset = CuVSMatrix.ofArray((float[][]) vectors);
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
    return writeGraph(graph, vectorIndex, 1);
  }

  static int[][] writeGraph(GPUBuiltHnswGraph graph, IndexOutput vectorIndex, int requestedWorkers)
      throws IOException {
    return writeGraph(
        graph, vectorIndex, requestedWorkers, Runtime.getRuntime().availableProcessors());
  }

  static int[][] writeGraph(
      GPUBuiltHnswGraph graph,
      IndexOutput vectorIndex,
      int requestedWorkers,
      int availableProcessors)
      throws IOException {
    int countOnLevel0 = graph.size();
    int numLevels = graph.numLevels();
    int[][] offsets = new int[numLevels][];
    int maxConn = graph.maxConn();

    int[] level0Nodes = NodesIterator.getSortedNodes(graph.getNodesOnLevel(0));
    offsets[0] = new int[level0Nodes.length];
    if (requestedWorkers > 1 && level0Nodes.length >= PARALLEL_MIN_NODES) {
      writeLevel0Parallel(
          graph,
          vectorIndex,
          level0Nodes,
          offsets[0],
          countOnLevel0,
          maxConn,
          requestedWorkers,
          availableProcessors);
    } else {
      writeLevelSerial(graph, vectorIndex, 0, level0Nodes, offsets[0], countOnLevel0, maxConn);
    }

    for (int level = 1; level < numLevels; level++) {
      int[] sortedNodes = NodesIterator.getSortedNodes(graph.getNodesOnLevel(level));
      offsets[level] = new int[sortedNodes.length];
      writeLevelSerial(
          graph, vectorIndex, level, sortedNodes, offsets[level], countOnLevel0, maxConn);
    }
    return offsets;
  }

  /** Node count below which parallel level-zero serialization costs more than it saves. */
  static final int PARALLEL_MIN_NODES = 1 << 16;

  /** Maximum encoded payload retained by one parallel wave before it is written to the index. */
  static final long MAX_PARALLEL_ENCODE_BYTES = 64L << 20;

  private static final int MAX_VINT_BYTES = 5;

  private static void writeLevelSerial(
      GPUBuiltHnswGraph graph,
      IndexOutput output,
      int level,
      int[] sortedNodes,
      int[] offsets,
      int countOnLevel0,
      int maxConn)
      throws IOException {
    int[] scratch = new int[maxConn * 2];
    for (int index = 0; index < sortedNodes.length; index++) {
      long start = output.getFilePointer();
      encodeNode(
          requireNeighbors(graph, level, sortedNodes[index]), scratch, output, countOnLevel0);
      offsets[index] = Math.toIntExact(output.getFilePointer() - start);
    }
  }

  private static void writeLevel0Parallel(
      GPUBuiltHnswGraph graph,
      IndexOutput output,
      int[] nodes,
      int[] offsets,
      int countOnLevel0,
      int maxConn,
      int requestedWorkers,
      int availableProcessors)
      throws IOException {
    try (BoundedParallelExecutor executor =
        BoundedParallelExecutor.create(requestedWorkers, nodes.length, availableProcessors)) {
      if (!executor.isParallel()) {
        writeLevelSerial(graph, output, 0, nodes, offsets, countOnLevel0, maxConn);
        return;
      }

      int waveSize = nodesPerSerializationWave(maxConn);
      for (int waveStart = 0; waveStart < nodes.length; ) {
        int waveEnd = (int) Math.min(nodes.length, (long) waveStart + waveSize);
        int nodesInWave = waveEnd - waveStart;
        int currentWaveStart = waveStart;
        ByteBuffersDataOutput[] buffers =
            new ByteBuffersDataOutput[executor.workerCountFor(nodesInWave)];

        executor.invokeRanges(
            nodesInWave,
            (taskIndex, start, end) -> {
              ByteBuffersDataOutput buffer = new ByteBuffersDataOutput();
              int[] scratch = new int[maxConn * 2];
              for (int relativeIndex = start; relativeIndex < end; relativeIndex++) {
                if ((relativeIndex & 0x3ff) == 0 && Thread.currentThread().isInterrupted()) {
                  throw new InterruptedException("graph serialization interrupted");
                }
                int nodeIndex = currentWaveStart + relativeIndex;
                long before = buffer.size();
                encodeNode(
                    requireNeighbors(graph, 0, nodes[nodeIndex]), scratch, buffer, countOnLevel0);
                offsets[nodeIndex] = Math.toIntExact(buffer.size() - before);
              }
              buffers[taskIndex] = buffer;
            });

        for (ByteBuffersDataOutput buffer : buffers) {
          buffer.copyTo(output);
        }
        waveStart = waveEnd;
      }
    }
  }

  static int nodesPerSerializationWave(int maxConn) {
    long maxBytesPerNode = (Math.max(0L, maxConn) + 1L) * MAX_VINT_BYTES;
    return Math.toIntExact(
        Math.max(1L, Math.min(Integer.MAX_VALUE, MAX_PARALLEL_ENCODE_BYTES / maxBytesPerNode)));
  }

  private static void encodeNode(
      NeighborArray neighbors, int[] scratch, DataOutput output, int countOnLevel0)
      throws IOException {
    int size = neighbors.size();
    if (size > scratch.length) {
      throw new IllegalArgumentException(
          "Neighbor count " + size + " exceeds scratch capacity " + scratch.length);
    }
    int actualSize = 0;
    if (size > 0) {
      int[] nodes = neighbors.nodes();
      Arrays.sort(nodes, 0, size);
      scratch[0] = nodes[0];
      actualSize = 1;
      for (int index = 1; index < size; index++) {
        assert nodes[index] < countOnLevel0
            : "node too large: " + nodes[index] + ">=" + countOnLevel0;
        if (nodes[index - 1] != nodes[index]) {
          scratch[actualSize++] = nodes[index] - nodes[index - 1];
        }
      }
    }
    output.writeVInt(actualSize);
    for (int index = 0; index < actualSize; index++) {
      output.writeVInt(scratch[index]);
    }
  }

  private static NeighborArray requireNeighbors(GPUBuiltHnswGraph graph, int level, int node)
      throws IOException {
    NeighborArray neighbors = graph.getNeighbors(level, node);
    if (neighbors == null) {
      throw new IOException("Missing neighbors for node " + node + " on level " + level);
    }
    return neighbors;
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
      int[][] graphLevelNodeOffsets)
      throws IOException {

    meta.writeInt(field.number);
    meta.writeInt(field.getVectorEncoding().ordinal());
    meta.writeInt(distFuncToOrd(field.getVectorSimilarityFunction()));
    meta.writeVLong(vectorIndexOffset);
    meta.writeVLong(vectorIndexLength);
    meta.writeVInt(field.getVectorDimension());
    meta.writeInt(count);
    // M = ceil(cagraGraphDegree / 2), derived from the graph being written rather than from a
    // caller-supplied degree: graph.maxConn() is the widest layer-0 adjacency row, which is the
    // degree cuVS actually built (it may truncate the requested one for small datasets).
    meta.writeVInt(graph == null ? 0 : Math.ceilDiv(graph.maxConn(), 2));

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
    writeMeta(null, op, fieldInfo, 0, 0, 0, null, null);
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

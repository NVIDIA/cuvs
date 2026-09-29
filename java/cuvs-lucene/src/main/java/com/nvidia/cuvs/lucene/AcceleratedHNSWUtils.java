/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.getCuVSResourcesInstance;
import static com.nvidia.cuvs.lucene.Utils.createHostByteMatrixFromArray;

import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.RowView;
import java.io.IOException;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.concurrent.Callable;
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

    // GPUBuiltHnswGraph copies the adjacency into heap-backed NeighborArrays.
    try (CuVSMatrix adjacencyMatrix = CuVSMatrix.ofArray(singleNodeAdjacency)) {
      List<int[]> layerNodes = new ArrayList<>();
      layerNodes.add(null); // Layer 0 contains all nodes, so its node list is implicit.
      return new GPUBuiltHnswGraph(size, dimensions, layerNodes, List.of(adjacencyMatrix));
    }
  }

  /**
   * Creates up to {@code hnswLayers} total layers. Layer 0 uses the full CAGRA graph. Each upper
   * layer samples {@code max(2, floor(previousLayerSize / M))} nodes. The value {@code M} is the
   * ceiling of half the layer-0 graph degree.
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
        size,
        dimensions,
        adjacencyListMatrix,
        vectors,
        hnswLayers,
        params,
        quantization,
        AcceleratedHNSWParams.DEFAULT_GRAPH_THREADS);
  }

  private static GPUBuiltHnswGraph createMultiLayerHnswGraph(
      int size,
      int dimensions,
      CuVSMatrix adjacencyListMatrix,
      List<?> vectors,
      int hnswLayers,
      CagraIndexParams params,
      QuantizationType quantization,
      int graphThreads)
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
    Throwable failure = null;

    try {
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

        CuVSMatrix upperAdjacency;
        if (quantization == QuantizationType.NONE) {
          // Extract vectors for selected nodes
          float[][] selectedVectors = new float[nextLayerSize][];
          for (int i = 0; i < nextLayerSize; i++) {
            selectedVectors[i] = (float[]) vectors.get(selectedNodes[i]);
          }

          // Build CAGRA graph for this layer
          upperAdjacency =
              buildCagraGraphForSubset(
                  selectedVectors, selectedNodes, 0, params, dimensions, quantization);

        } else {

          // Extract vectors for selected nodes
          int bytesPerVector = (dimensions + 7) / 8;
          byte[][] selectedVectors = new byte[nextLayerSize][];
          for (int i = 0; i < nextLayerSize; i++) {
            selectedVectors[i] = (byte[]) vectors.get(selectedNodes[i]);
          }

          // Build CAGRA graph for this layer
          upperAdjacency =
              buildCagraGraphForSubset(
                  selectedVectors, selectedNodes, bytesPerVector, params, dimensions, quantization);
        }

        try {
          // Register ownership before any later operation can fail.
          layerAdjacencies.add(upperAdjacency);
        } catch (Throwable registrationFailure) {
          closeAfterFailure(upperAdjacency, registrationFailure);
          throw registrationFailure;
        }
        layerNodes.add(selectedNodes);

        // Update for next iteration
        currentLayerSize = nextLayerSize;
        layerIndex++;

        // Use different seed for each layer
        random = new Random(new Random().nextLong());
      }

      // The graph eagerly copies all adjacency rows, so generated upper matrices can now close.
      return new GPUBuiltHnswGraph(size, dimensions, layerNodes, layerAdjacencies, graphThreads);
    } catch (Throwable t) {
      failure = t;
      throw t;
    } finally {
      Throwable closeFailure = closeUpperLayerAdjacencies(layerAdjacencies);
      if (closeFailure != null) {
        if (failure == null) {
          throw closeFailure;
        }
        if (failure != closeFailure) {
          failure.addSuppressed(closeFailure);
        }
      }
    }
  }

  /**
   * Creates a multi-layer HNSW graph from a native matrix without copying the complete dataset to
   * the Java heap. The list view copies only rows selected for an upper layer.
   */
  static GPUBuiltHnswGraph createMultiLayerHnswGraph(
      int dimensions,
      CuVSMatrix adjacencyListMatrix,
      CuVSMatrix vectorDataset,
      int hnswLayers,
      CagraIndexParams params,
      QuantizationType quantization)
      throws Throwable {
    return createMultiLayerHnswGraph(
        dimensions,
        adjacencyListMatrix,
        vectorDataset,
        hnswLayers,
        params,
        quantization,
        AcceleratedHNSWParams.DEFAULT_GRAPH_THREADS);
  }

  static GPUBuiltHnswGraph createMultiLayerHnswGraph(
      int dimensions,
      CuVSMatrix adjacencyListMatrix,
      CuVSMatrix vectorDataset,
      int hnswLayers,
      CagraIndexParams params,
      QuantizationType quantization,
      int graphThreads)
      throws Throwable {
    int size = Math.toIntExact(vectorDataset.size());
    // Matrix columns are the stored width: binary vectors are bit-packed, while scalar and float
    // vectors store one value per dimension.
    int columns = Math.toIntExact(vectorDataset.columns());
    List<?> vectors =
        new AbstractList<>() {
          @Override
          public Object get(int index) {
            RowView row = vectorDataset.getRow(index);
            if (quantization == QuantizationType.NONE) {
              float[] vector = new float[columns];
              row.toArray(vector);
              return vector;
            }
            byte[] vector = new byte[columns];
            row.toArray(vector);
            return vector;
          }

          @Override
          public int size() {
            return size;
          }
        };
    return createMultiLayerHnswGraph(
        size,
        dimensions,
        adjacencyListMatrix,
        vectors,
        hnswLayers,
        params,
        quantization,
        graphThreads);
  }

  private static Throwable closeUpperLayerAdjacencies(List<CuVSMatrix> layerAdjacencies) {
    Throwable failure = null;
    // Layer 0 is borrowed from the outer CAGRA index. Only upper layers are owned here.
    for (int i = layerAdjacencies.size() - 1; i >= 1; i--) {
      try {
        layerAdjacencies.get(i).close();
      } catch (Throwable closeFailure) {
        if (failure == null) {
          failure = closeFailure;
        } else if (failure != closeFailure) {
          failure.addSuppressed(closeFailure);
        }
      }
    }
    return failure;
  }

  private static void closeAfterFailure(AutoCloseable resource, Throwable failure) {
    try {
      resource.close();
    } catch (Throwable closeFailure) {
      if (failure != closeFailure) {
        failure.addSuppressed(closeFailure);
      }
    }
  }

  /** Builds a CAGRA graph for a selected vector subset. */
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
      subsetDataset = createHostByteMatrixFromArray((byte[][]) vectors, bytesPerVector);
    } else if (quantization == QuantizationType.SCALAR) {
      subsetDataset = createHostByteMatrixFromArray((byte[][]) vectors, dimensions);
    } else {
      subsetDataset = CuVSMatrix.ofArray((float[][]) vectors);
    }

    return buildCagraGraphForSubset(subsetDataset, selectedNodes, params);
  }

  private static CuVSMatrix buildCagraGraphForSubset(
      CuVSMatrix subsetDataset, int[] selectedNodes, CagraIndexParams params) throws Throwable {
    int[][] remappedAdjacency;
    try (Utils.OwnedIndex<CagraIndex> ownedIndex = Utils.ownDataset(subsetDataset)) {
      CagraIndex subsetIndex =
          CagraIndex.newBuilder(getCuVSResourcesInstance())
              .withDataset(subsetDataset)
              .withIndexParams(params)
              .build();
      ownedIndex.transferTo(subsetIndex);

      CuVSMatrix cagraGraph = subsetIndex.getGraph();
      long numNodes = cagraGraph.size();
      long degree = cagraGraph.columns();
      remappedAdjacency = new int[(int) numNodes][(int) degree];

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
    }

    // Build the returned matrix only after the subset index and its dataset have closed.
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
    return writeGraph(graph, vectorIndex, AcceleratedHNSWParams.DEFAULT_GRAPH_THREADS);
  }

  static int[][] writeGraph(GPUBuiltHnswGraph graph, IndexOutput vectorIndex, int graphThreads)
      throws IOException {
    int countOnLevel0 = graph.size();
    int numLevels = graph.numLevels();
    int[][] offsets = new int[numLevels][];
    int maxConn = graph.maxConn();

    int[] level0Nodes = NodesIterator.getSortedNodes(graph.getNodesOnLevel(0));
    offsets[0] = new int[level0Nodes.length];
    if (graphThreads > 1 && level0Nodes.length >= GPUBuiltHnswGraph.PARALLEL_MIN_NODES) {
      writeLevel0Parallel(
          graph, vectorIndex, level0Nodes, offsets[0], countOnLevel0, maxConn, graphThreads);
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

  /** Absolute node cap for one serialization wave. */
  static final int MAX_SERIALIZATION_WAVE_NODES = 1 << 20;

  /** Maximum worst-case encoded payload buffered by one serialization wave. */
  static final long MAX_SERIALIZED_BYTES_PER_WAVE = 64L << 20;

  private static final int MAX_VINT_BYTES = 5;

  private static void writeLevelSerial(
      GPUBuiltHnswGraph graph,
      IndexOutput out,
      int level,
      int[] nodes,
      int[] offsets,
      int countOnLevel0,
      int maxConn)
      throws IOException {
    int[] scratch = new int[maxConn * 2];
    for (int i = 0; i < nodes.length; i++) {
      long start = out.getFilePointer();
      encodeNode(graph.getNeighbors(level, nodes[i]), scratch, out, countOnLevel0);
      offsets[i] = Math.toIntExact(out.getFilePointer() - start);
    }
  }

  /** Encodes level zero in bounded waves, then concatenates buffers in node order. */
  private static void writeLevel0Parallel(
      GPUBuiltHnswGraph graph,
      IndexOutput out,
      int[] nodes,
      int[] offsets,
      int countOnLevel0,
      int maxConn,
      int graphThreads)
      throws IOException {
    int waveNodes = serializationWaveNodes(maxConn);
    for (int waveStart = 0; waveStart < nodes.length; ) {
      int waveEnd = (int) Math.min(nodes.length, (long) waveStart + waveNodes);
      int nodesPerTask = Math.ceilDiv(waveEnd - waveStart, graphThreads);
      ByteBuffersDataOutput[] buffers = new ByteBuffersDataOutput[graphThreads];
      List<Callable<Void>> tasks = new ArrayList<>(graphThreads);
      for (int task = 0; task < graphThreads; task++) {
        int start = waveStart + task * nodesPerTask;
        int end = Math.min(start + nodesPerTask, waveEnd);
        int bufferIndex = task;
        if (start >= end) {
          break;
        }
        tasks.add(
            () -> {
              ByteBuffersDataOutput buffer = new ByteBuffersDataOutput();
              int[] scratch = new int[maxConn * 2];
              for (int i = start; i < end; i++) {
                long before = buffer.size();
                encodeNode(graph.getNeighbors(0, nodes[i]), scratch, buffer, countOnLevel0);
                offsets[i] = Math.toIntExact(buffer.size() - before);
              }
              buffers[bufferIndex] = buffer;
              return null;
            });
      }
      GraphWorkExecutor.invokeAll(tasks);
      for (ByteBuffersDataOutput buffer : buffers) {
        if (buffer != null) {
          buffer.copyTo(out);
        }
      }
      waveStart = waveEnd;
    }
  }

  static int serializationWaveNodes(int maxConn) {
    if (maxConn < 0) {
      throw new IllegalArgumentException("maxConn must not be negative");
    }
    long maxBytesPerNode = Math.addExact(MAX_VINT_BYTES, (long) maxConn * MAX_VINT_BYTES);
    long byteBoundedNodes = Math.max(1, MAX_SERIALIZED_BYTES_PER_WAVE / maxBytesPerNode);
    return (int) Math.min(MAX_SERIALIZATION_WAVE_NODES, byteBoundedNodes);
  }

  private static void encodeNode(
      NeighborArray neighbors, int[] scratch, DataOutput out, int countOnLevel0)
      throws IOException {
    int size = neighbors == null ? 0 : neighbors.size();
    int actualSize = 0;
    if (size > 0) {
      int[] nodes = neighbors.nodes();
      Arrays.sort(nodes, 0, size);
      scratch[0] = nodes[0];
      actualSize = 1;
      for (int i = 1; i < size; i++) {
        assert nodes[i] < countOnLevel0 : "node too large: " + nodes[i] + ">=" + countOnLevel0;
        if (nodes[i - 1] != nodes[i]) {
          scratch[actualSize++] = nodes[i] - nodes[i - 1];
        }
      }
    }
    out.writeVInt(actualSize);
    for (int i = 0; i < actualSize; i++) {
      out.writeVInt(scratch[i]);
    }
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

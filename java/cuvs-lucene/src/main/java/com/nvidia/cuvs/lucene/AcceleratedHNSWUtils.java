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
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
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
    return new GPUBuiltHnswGraph(size, dimensions, layerNodes, layerAdjacencies, 1);
  }

  /**
   * Creates a multi-layer HNSW graph with dynamic number of layers.
   * M = ceil(cagraGraphDegree / 2), where cagraGraphDegree is the CAGRA adjacency list's degree
   * (its column count). Ceil is used to accommodate odd graph degrees.
   * Each layer contains 1/M nodes from the previous layer
   * Creates layers until the highest layer has <= M nodes
   * <p>
   * Vectors for higher-layer subsets are read directly from the native matrix
   * via {@link CuVSMatrix#getRow(long)} and {@link RowView#toArray(float[])},
   * avoiding any additional heap allocation of the full dataset. Used by both
   * the flush and merge paths; the caller provides the vectors as a
   * {@link CuVSMatrix}.
   */
  public static GPUBuiltHnswGraph createMultiLayerHnswGraph(
      FieldInfo fieldInfo,
      int dimensions,
      CuVSMatrix adjacencyListMatrix,
      CuVSMatrix vectorDataset,
      int hnswLayers,
      int graphDegree,
      CagraIndexParams params,
      QuantizationType quantization,
      int numThreads)
      throws Throwable {

    int size = (int) vectorDataset.size();
    int M = graphDegree / 2;

    List<int[]> layerNodes = new ArrayList<>();
    List<CuVSMatrix> layerAdjacencies = new ArrayList<>();

    // Layer 0: Use full CAGRA adjacency list
    layerNodes.add(null);
    layerAdjacencies.add(adjacencyListMatrix);

    int currentLayerSize = size;
    int layerIndex = 1;
    Random random = new Random();

    while (layerIndex < hnswLayers && currentLayerSize > 1) {
      int nextLayerSize = Math.max(2, currentLayerSize / M);
      SortedSet<Integer> selectedNodesSet = new TreeSet<>();

      if (layerIndex == 1) {
        while (selectedNodesSet.size() < nextLayerSize) {
          selectedNodesSet.add(random.nextInt(size));
        }
      } else {
        int[] prevLayerNodes = layerNodes.get(layerNodes.size() - 1);
        while (selectedNodesSet.size() < nextLayerSize) {
          selectedNodesSet.add(prevLayerNodes[random.nextInt(prevLayerNodes.length)]);
        }
      }

      int[] selectedNodes =
          selectedNodesSet.stream().mapToInt(Integer::intValue).sorted().toArray();
      layerNodes.add(selectedNodes);

      if (quantization == QuantizationType.NONE) {
        // Read only the sampled rows from the native matrix — no full-dataset heap copy
        float[][] selectedVectors = new float[nextLayerSize][dimensions];
        for (int i = 0; i < nextLayerSize; i++) {
          vectorDataset.getRow(selectedNodes[i]).toArray(selectedVectors[i]);
        }
        layerAdjacencies.add(
            buildCagraGraphForSubset(
                selectedVectors, selectedNodes, 0, params, dimensions, quantization));
      } else {
        // Byte width comes from the matrix itself: binary packs 8 dims/byte, scalar is 1 byte/dim.
        int bytesPerVector = (int) vectorDataset.columns();
        byte[][] selectedVectors = new byte[nextLayerSize][bytesPerVector];
        for (int i = 0; i < nextLayerSize; i++) {
          vectorDataset.getRow(selectedNodes[i]).toArray(selectedVectors[i]);
        }
        layerAdjacencies.add(
            buildCagraGraphForSubset(
                selectedVectors, selectedNodes, bytesPerVector, params, dimensions, quantization));
      }

      currentLayerSize = nextLayerSize;
      layerIndex++;
      random = new Random(new Random().nextLong());
    }

    return new GPUBuiltHnswGraph(size, dimensions, layerNodes, layerAdjacencies, numThreads);
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
      subsetDataset = createByteMatrixFromArray((byte[][]) vectors, bytesPerVector);
    } else if (quantization == QuantizationType.SCALAR) {
      subsetDataset = createByteMatrixFromArray((byte[][]) vectors, dimensions);
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
  public static int[][] writeGraph(
      GPUBuiltHnswGraph graph, IndexOutput vectorIndex, int numThreads) throws IOException {
    int countOnLevel0 = graph.size();
    int numLevels = graph.numLevels();
    int[][] offsets = new int[numLevels][];

    // Level 0 holds all nodes and dominates serialization cost. Each node's delta/VInt block is
    // independent, so encode level 0 in parallel and concatenate the per-thread buffers serially in
    // node order, in memory-bounded waves. Higher levels are tiny and stay serial. The on-disk bytes
    // are identical to the fully-serial path (blocks in node order, offsets = per-node byte lengths).
    int[] level0Nodes = NodesIterator.getSortedNodes(graph.getNodesOnLevel(0));
    offsets[0] = new int[level0Nodes.length];
    if (numThreads > 1 && level0Nodes.length >= PARALLEL_MIN_NODES) {
      writeLevel0Parallel(graph, vectorIndex, level0Nodes, offsets[0], countOnLevel0, numThreads);
    } else {
      writeLevelSerial(graph, vectorIndex, 0, level0Nodes, offsets[0], countOnLevel0);
    }

    for (int level = 1; level < numLevels; level++) {
      int[] sortedNodes = NodesIterator.getSortedNodes(graph.getNodesOnLevel(level));
      offsets[level] = new int[sortedNodes.length];
      writeLevelSerial(graph, vectorIndex, level, sortedNodes, offsets[level], countOnLevel0);
    }
    return offsets;
  }

  /** Node count below which parallel level-0 serialization is not worth the overhead. */
  private static final int PARALLEL_MIN_NODES = 1 << 16;

  /** Nodes per wave — bounds the transient encode buffer regardless of dataset size. */
  private static final int WAVE_NODES = 1 << 20;

  /** Serially encodes a level's nodes into {@code out}, recording per-node byte lengths. */
  private static void writeLevelSerial(
      GPUBuiltHnswGraph graph,
      IndexOutput out,
      int level,
      int[] sortedNodes,
      int[] offsets,
      int countOnLevel0)
      throws IOException {
    int[] scratch = new int[graph.maxConn() * 2];
    int idx = 0;
    for (int node : sortedNodes) {
      long start = out.getFilePointer();
      encodeNode(graph.getNeighbors(level, node), scratch, out, countOnLevel0);
      offsets[idx++] = Math.toIntExact(out.getFilePointer() - start);
    }
  }

  /**
   * Encodes level 0 in parallel: within memory-bounded waves, threads encode contiguous node
   * sub-ranges into per-thread buffers, which are then concatenated to {@code out} in node order
   * (identical layout to the serial path).
   */
  private static void writeLevel0Parallel(
      GPUBuiltHnswGraph graph,
      IndexOutput out,
      int[] nodes,
      int[] offsets,
      int countOnLevel0,
      int numThreads)
      throws IOException {
    ExecutorService pool = Executors.newFixedThreadPool(numThreads);
    try {
      int n = nodes.length;
      for (int waveStart = 0; waveStart < n; waveStart += WAVE_NODES) {
        int waveEnd = Math.min(waveStart + WAVE_NODES, n);
        int perThread = (waveEnd - waveStart + numThreads - 1) / numThreads;

        ByteBuffersDataOutput[] buffers = new ByteBuffersDataOutput[numThreads];
        List<Future<?>> futures = new ArrayList<>(numThreads);
        for (int t = 0; t < numThreads; t++) {
          final int subStart = waveStart + t * perThread;
          final int subEnd = Math.min(subStart + perThread, waveEnd);
          final int slot = t;
          if (subStart >= subEnd) {
            continue;
          }
          futures.add(
              pool.submit(
                  () -> {
                    ByteBuffersDataOutput buffer = new ByteBuffersDataOutput();
                    int[] scratch = new int[graph.maxConn() * 2];
                    for (int i = subStart; i < subEnd; i++) {
                      long before = buffer.size();
                      encodeNode(graph.getNeighbors(0, nodes[i]), scratch, buffer, countOnLevel0);
                      offsets[i] = Math.toIntExact(buffer.size() - before);
                    }
                    buffers[slot] = buffer;
                    return null;
                  }));
        }
        for (Future<?> f : futures) {
          f.get();
        }
        // Concatenate in thread order (== node order), preserving the serial byte layout.
        for (ByteBuffersDataOutput buffer : buffers) {
          if (buffer != null) {
            buffer.copyTo(out);
          }
        }
      }
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IOException("Interrupted during parallel writeGraph", e);
    } catch (ExecutionException e) {
      throw new IOException("Parallel writeGraph failed", e.getCause());
    } finally {
      pool.shutdown();
    }
  }

  /**
   * Sorts, delta-encodes and de-duplicates a node's neighbors and writes the block (VInt size + VInt
   * deltas) to {@code out}. Shared by the serial and parallel paths so encoding is identical.
   */
  private static void encodeNode(
      NeighborArray neighbors, int[] scratch, DataOutput out, int countOnLevel0) throws IOException {
    int size = neighbors == null ? 0 : neighbors.size();
    int actualSize = 0;
    if (size > 0) {
      int[] nnodes = neighbors.nodes();
      Arrays.sort(nnodes, 0, size);
      scratch[0] = nnodes[0];
      actualSize = 1;
      for (int i = 1; i < size; i++) {
        assert nnodes[i] < countOnLevel0 : "node too large: " + nnodes[i] + ">=" + countOnLevel0;
        if (nnodes[i - 1] == nnodes[i]) {
          continue;
        }
        scratch[actualSize++] = nnodes[i] - nnodes[i - 1];
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

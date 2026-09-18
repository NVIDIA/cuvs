/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.getCuVSResourcesInstance;

import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.RowView;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
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
    try (CuVSMatrix adjacencyMatrix = CuVSMatrix.ofArray(singleNodeAdjacency)) {
      // Layer 0 contains every node, so its node list is implicit.
      List<int[]> layerNodes = new ArrayList<>();
      layerNodes.add(null);
      return new GPUBuiltHnswGraph(size, dimensions, layerNodes, List.of(adjacencyMatrix));
    }
  }

  /**
   * Creates a multi-layer HNSW graph from heap vectors.
   *
   * <p>Layer 0 retains the complete CAGRA graph. Let {@code M = ceil(degree / 2)}, where the degree
   * is the layer-0 adjacency width. Each requested upper layer contains at least two nodes and
   * otherwise uses {@code floor(previousLayerSize / M)} distinct nodes sampled from the preceding
   * layer. Construction is bounded by {@code hnswLayers} and stops early when the preceding layer
   * has at most one node. Only rows selected for an upper layer are copied into native memory.
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
    Objects.requireNonNull(vectors, "vectors");
    if (vectors.size() != size) {
      throw new IllegalArgumentException(
          "Expected " + size + " vectors, but received " + vectors.size());
    }
    CuVSMatrix.DataType dataType = expectedDataType(quantization);
    int columns = expectedColumns(dimensions, quantization);
    return createMultiLayerHnswGraph(
        size,
        dimensions,
        adjacencyListMatrix,
        hnswLayers,
        params,
        selectedNodes -> createSubsetDataset(vectors, selectedNodes, columns, dataType));
  }

  /*
   * Creates a multi-layer HNSW graph from a native matrix.
   *
   * <p>Only sampled rows are copied into each upper-layer matrix; the complete dataset is never
   * materialized on the Java heap.
   */
  static GPUBuiltHnswGraph createMultiLayerHnswGraph(
      int dimensions,
      CuVSMatrix adjacencyListMatrix,
      CuVSMatrix vectorDataset,
      int hnswLayers,
      CagraIndexParams params,
      QuantizationType quantization)
      throws Throwable {
    Objects.requireNonNull(vectorDataset, "vectorDataset");
    int size = Math.toIntExact(vectorDataset.size());
    CuVSMatrix.DataType dataType = expectedDataType(quantization);
    int columns = expectedColumns(dimensions, quantization);
    if (vectorDataset.columns() != columns) {
      throw new IllegalArgumentException(
          "Expected " + columns + " matrix columns, but received " + vectorDataset.columns());
    }
    if (vectorDataset.dataType() != dataType) {
      throw new IllegalArgumentException(
          "Expected " + dataType + " matrix data, but received " + vectorDataset.dataType());
    }
    return createMultiLayerHnswGraph(
        size,
        dimensions,
        adjacencyListMatrix,
        hnswLayers,
        params,
        selectedNodes -> createSubsetDataset(vectorDataset, selectedNodes, columns, dataType));
  }

  private static GPUBuiltHnswGraph createMultiLayerHnswGraph(
      int size,
      int dimensions,
      CuVSMatrix adjacencyListMatrix,
      int hnswLayers,
      CagraIndexParams params,
      SubsetDatasetFactory subsetDatasetFactory)
      throws Throwable {
    Objects.requireNonNull(adjacencyListMatrix, "adjacencyListMatrix");
    Objects.requireNonNull(params, "params");
    if (size < 2) {
      throw new IllegalArgumentException("A multi-layer graph requires at least two vectors");
    }
    if (adjacencyListMatrix.dataType() != CuVSMatrix.DataType.INT
        && adjacencyListMatrix.dataType() != CuVSMatrix.DataType.UINT) {
      throw new IllegalArgumentException(
          "Expected INT or UINT adjacency data, but received " + adjacencyListMatrix.dataType());
    }
    if (adjacencyListMatrix.size() != size) {
      throw new IllegalArgumentException(
          "Expected " + size + " adjacency rows, but received " + adjacencyListMatrix.size());
    }

    int degree = Math.toIntExact(adjacencyListMatrix.columns());
    if (degree <= 0) {
      throw new IllegalArgumentException("The layer-0 graph must have a positive degree");
    }
    int M = Math.ceilDiv(degree, 2);

    List<int[]> layerNodes = new ArrayList<>();
    List<CuVSMatrix> layerAdjacencies = new ArrayList<>();

    // Layer 0 retains the complete CAGRA adjacency list.
    layerNodes.add(null);
    layerAdjacencies.add(adjacencyListMatrix);

    int currentLayerSize = size;
    int layerIndex = 1;
    Random random = new Random();

    Throwable failure = null;
    try {
      while (layerIndex < hnswLayers && currentLayerSize > 1) {
        // Each upper layer samples roughly 1/M of the preceding layer, with a two-node floor.
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
        CuVSMatrix subsetDataset = subsetDatasetFactory.create(selectedNodes);
        CuVSMatrix upperAdjacency = buildCagraGraphForSubset(subsetDataset, selectedNodes, params);
        try {
          layerNodes.add(selectedNodes);
          layerAdjacencies.add(upperAdjacency);
        } catch (Throwable addFailure) {
          try {
            upperAdjacency.close();
          } catch (Throwable closeFailure) {
            if (addFailure != closeFailure) {
              addFailure.addSuppressed(closeFailure);
            }
          }
          throw addFailure;
        }

        currentLayerSize = nextLayerSize;
        layerIndex++;
        random = new Random(new Random().nextLong());
      }

      return new GPUBuiltHnswGraph(size, dimensions, layerNodes, layerAdjacencies);
    } catch (Throwable t) {
      failure = t;
      throw t;
    } finally {
      Throwable closeFailure = null;
      for (int i = layerAdjacencies.size() - 1; i >= 1; i--) {
        try {
          layerAdjacencies.get(i).close();
        } catch (Throwable t) {
          if (closeFailure == null) {
            closeFailure = t;
          } else if (closeFailure != t) {
            closeFailure.addSuppressed(t);
          }
        }
      }
      if (closeFailure != null) {
        if (failure != null && failure != closeFailure) {
          failure.addSuppressed(closeFailure);
        } else {
          throw closeFailure;
        }
      }
    }
  }

  /** Builds a CAGRA graph for a sampled subset and remaps its ordinals to layer-0 ordinals. */
  private static CuVSMatrix buildCagraGraphForSubset(
      CuVSMatrix subsetDataset, int[] selectedNodes, CagraIndexParams params) throws Throwable {
    return buildCagraGraphForSubset(
        subsetDataset,
        selectedNodes,
        params,
        (dataset, indexParams) ->
            CagraIndex.newBuilder(getCuVSResourcesInstance())
                .withDataset(dataset)
                .withIndexParams(indexParams)
                .build());
  }

  static CuVSMatrix buildCagraGraphForSubset(
      CuVSMatrix subsetDataset,
      int[] selectedNodes,
      CagraIndexParams params,
      SubsetIndexFactory indexFactory)
      throws Throwable {
    CuVSMatrix remappedAdjacency = null;
    try (Utils.OwnedIndex<CagraIndex> ownedIndex = Utils.ownDataset(subsetDataset)) {
      CagraIndex subsetIndex = indexFactory.build(subsetDataset, params);
      ownedIndex.transferTo(subsetIndex);
      remappedAdjacency = remapSubsetGraph(subsetIndex.getGraph(), selectedNodes);
    } catch (Throwable failure) {
      if (remappedAdjacency != null) {
        try {
          remappedAdjacency.close();
        } catch (Throwable closeFailure) {
          if (failure != closeFailure) {
            failure.addSuppressed(closeFailure);
          }
        }
      }
      throw failure;
    }
    return remappedAdjacency;
  }

  static CuVSMatrix remapSubsetGraph(CuVSMatrix cagraGraph, int[] selectedNodes)
      throws IOException {
    if (cagraGraph.dataType() != CuVSMatrix.DataType.INT
        && cagraGraph.dataType() != CuVSMatrix.DataType.UINT) {
      throw new IOException(
          "Expected INT or UINT subset adjacency data, but received " + cagraGraph.dataType());
    }
    if (cagraGraph.size() != selectedNodes.length) {
      throw new IOException(
          "Expected "
              + selectedNodes.length
              + " subset adjacency rows, but received "
              + cagraGraph.size());
    }
    int degree = Math.toIntExact(cagraGraph.columns());
    if (degree <= 0) {
      throw new IOException("The subset graph must have a positive degree");
    }
    int[] remappedRow = new int[degree];
    return MatrixBuilderLifecycle.build(
        CuVSMatrix.hostBuilder(selectedNodes.length, degree, CuVSMatrix.DataType.INT),
        builder -> {
          for (int rowOrdinal = 0; rowOrdinal < selectedNodes.length; rowOrdinal++) {
            RowView row = cagraGraph.getRow(rowOrdinal);
            remapSubsetAdjacencyRow(rowOrdinal, row, degree, selectedNodes, remappedRow);
            builder.addVector(remappedRow);
          }
          return builder.build();
        });
  }

  static void remapSubsetAdjacencyRow(
      int rowOrdinal, RowView source, int degree, int[] selectedNodes, int[] destination)
      throws IOException {
    if (source == null || source.size() != degree) {
      throw new IOException(
          "Expected "
              + degree
              + " neighbors for subset row "
              + rowOrdinal
              + ", but received "
              + (source == null ? "null" : source.size()));
    }
    for (int column = 0; column < degree; column++) {
      int subsetOrdinal = source.getAsInt(column);
      if (subsetOrdinal < 0) {
        destination[column] = subsetOrdinal;
      } else if (subsetOrdinal >= selectedNodes.length) {
        throw new IOException(
            "Subset adjacency row "
                + rowOrdinal
                + ", column "
                + column
                + " contains local ordinal "
                + subsetOrdinal
                + " outside [0, "
                + selectedNodes.length
                + ")");
      } else {
        destination[column] = selectedNodes[subsetOrdinal];
      }
    }
  }

  private static CuVSMatrix createSubsetDataset(
      CuVSMatrix vectors, int[] selectedNodes, int columns, CuVSMatrix.DataType dataType)
      throws IOException {
    return MatrixBuilderLifecycle.build(
        CuVSMatrix.hostBuilder(selectedNodes.length, columns, dataType),
        builder -> {
          if (dataType == CuVSMatrix.DataType.FLOAT) {
            float[] rowBuffer = new float[columns];
            for (int node : selectedNodes) {
              RowView row = vectors.getRow(node);
              validateRowWidth(row, columns, node);
              row.toArray(rowBuffer);
              builder.addVector(rowBuffer);
            }
          } else {
            byte[] rowBuffer = new byte[columns];
            for (int node : selectedNodes) {
              RowView row = vectors.getRow(node);
              validateRowWidth(row, columns, node);
              row.toArray(rowBuffer);
              builder.addVector(rowBuffer);
            }
          }
          return builder.build();
        });
  }

  private static CuVSMatrix createSubsetDataset(
      List<?> vectors, int[] selectedNodes, int columns, CuVSMatrix.DataType dataType)
      throws IOException {
    return MatrixBuilderLifecycle.build(
        CuVSMatrix.hostBuilder(selectedNodes.length, columns, dataType),
        builder -> {
          for (int node : selectedNodes) {
            Object vector = vectors.get(node);
            if (dataType == CuVSMatrix.DataType.FLOAT) {
              if (!(vector instanceof float[] values) || values.length != columns) {
                throw new IllegalArgumentException(
                    "Vector " + node + " must be a float[" + columns + "]");
              }
              builder.addVector(values);
            } else {
              if (!(vector instanceof byte[] values) || values.length != columns) {
                throw new IllegalArgumentException(
                    "Vector " + node + " must be a byte[" + columns + "]");
              }
              builder.addVector(values);
            }
          }
          return builder.build();
        });
  }

  private static void validateRowWidth(RowView row, int columns, int rowIndex) throws IOException {
    if (row == null || row.size() != columns) {
      throw new IOException(
          "Expected "
              + columns
              + " values for vector row "
              + rowIndex
              + ", but received "
              + (row == null ? "null" : row.size()));
    }
  }

  private static int expectedColumns(int dimensions, QuantizationType quantization) {
    if (dimensions <= 0) {
      throw new IllegalArgumentException("Vector dimensions must be positive");
    }
    return quantization == QuantizationType.BINARY ? Math.ceilDiv(dimensions, 8) : dimensions;
  }

  private static CuVSMatrix.DataType expectedDataType(QuantizationType quantization) {
    Objects.requireNonNull(quantization, "quantization");
    return quantization == QuantizationType.NONE
        ? CuVSMatrix.DataType.FLOAT
        : CuVSMatrix.DataType.BYTE;
  }

  @FunctionalInterface
  private interface SubsetDatasetFactory {
    CuVSMatrix create(int[] selectedNodes) throws Throwable;
  }

  @FunctionalInterface
  interface SubsetIndexFactory {
    CagraIndex build(CuVSMatrix dataset, CagraIndexParams params) throws Throwable;
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
    BinaryQuantizer quantizer = new BinaryQuantizer(dimensions);
    for (float[] vector : floatVectors) {
      quantizer.add(vector);
    }
    quantizer.finish();

    List<byte[]> quantizedVectors = new ArrayList<>(floatVectors.size());
    for (float[] vector : floatVectors) {
      byte[] quantized = new byte[Math.ceilDiv(dimensions, 8)];
      quantizer.quantize(vector, quantized);
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
    ScalarQuantizer quantizer = new ScalarQuantizer(dimensions);
    for (float[] vector : floatVectors) {
      quantizer.add(vector);
    }
    quantizer.finish();

    List<byte[]> quantizedVectors = new ArrayList<>(floatVectors.size());
    for (float[] vector : floatVectors) {
      byte[] quantized = new byte[dimensions];
      quantizer.quantize(vector, quantized);
      quantizedVectors.add(quantized);
    }

    return quantizedVectors;
  }

  static final class BinaryQuantizer {
    private final float[] centroids;
    private int count;
    private boolean finished;

    BinaryQuantizer(int dimensions) {
      if (dimensions <= 0) {
        throw new IllegalArgumentException("Vector dimensions must be positive");
      }
      centroids = new float[dimensions];
    }

    void add(float[] vector) {
      ensureCollecting();
      requireDimensions(vector, centroids.length);
      for (int dimension = 0; dimension < vector.length; dimension++) {
        centroids[dimension] += vector[dimension];
      }
      count = Math.incrementExact(count);
    }

    void finish() {
      ensureCollecting();
      if (count == 0) {
        throw new IllegalStateException("Cannot finish binary quantization without vectors");
      }
      for (int dimension = 0; dimension < centroids.length; dimension++) {
        centroids[dimension] /= count;
      }
      finished = true;
    }

    void quantize(float[] vector, byte[] destination) {
      ensureFinished();
      requireDimensions(vector, centroids.length);
      if (destination.length != Math.ceilDiv(centroids.length, 8)) {
        throw new IllegalArgumentException("Binary quantization buffer dimensions do not match");
      }
      Arrays.fill(destination, (byte) 0);
      for (int dimension = 0; dimension < vector.length; dimension++) {
        if (vector[dimension] > centroids[dimension]) {
          destination[dimension / 8] |= (byte) (1 << (dimension % 8));
        }
      }
    }

    int count() {
      return count;
    }

    private void ensureCollecting() {
      if (finished) {
        throw new IllegalStateException("Quantization statistics are already final");
      }
    }

    private void ensureFinished() {
      if (finished == false) {
        throw new IllegalStateException("Quantization statistics are not final");
      }
    }
  }

  static final class ScalarQuantizer {
    private final float[] minima;
    private final float[] maxima;
    private int count;
    private boolean finished;

    ScalarQuantizer(int dimensions) {
      if (dimensions <= 0) {
        throw new IllegalArgumentException("Vector dimensions must be positive");
      }
      minima = new float[dimensions];
      maxima = new float[dimensions];
      Arrays.fill(minima, Float.MAX_VALUE);
      // Preserve the existing scalar quantizer's behavior for all-negative dimensions.
      Arrays.fill(maxima, Float.MIN_VALUE);
    }

    void add(float[] vector) {
      ensureCollecting();
      requireDimensions(vector, minima.length);
      for (int dimension = 0; dimension < vector.length; dimension++) {
        minima[dimension] = Math.min(minima[dimension], vector[dimension]);
        maxima[dimension] = Math.max(maxima[dimension], vector[dimension]);
      }
      count = Math.incrementExact(count);
    }

    void finish() {
      ensureCollecting();
      if (count == 0) {
        throw new IllegalStateException("Cannot finish scalar quantization without vectors");
      }
      finished = true;
    }

    void quantize(float[] vector, byte[] destination) {
      ensureFinished();
      requireDimensions(vector, minima.length);
      if (destination.length != minima.length) {
        throw new IllegalArgumentException("Scalar quantization buffer dimensions do not match");
      }
      for (int dimension = 0; dimension < vector.length; dimension++) {
        float range = maxima[dimension] - minima[dimension];
        if (range > 0) {
          float normalized = (vector[dimension] - minima[dimension]) / range;
          int quantizedValue = Math.round(normalized * 127.0f) - 64;
          destination[dimension] = (byte) Math.max(-64, Math.min(63, quantizedValue));
        } else {
          destination[dimension] = 0;
        }
      }
    }

    int count() {
      return count;
    }

    private void ensureCollecting() {
      if (finished) {
        throw new IllegalStateException("Quantization statistics are already final");
      }
    }

    private void ensureFinished() {
      if (finished == false) {
        throw new IllegalStateException("Quantization statistics are not final");
      }
    }
  }

  private static void requireDimensions(float[] vector, int dimensions) {
    if (vector.length != dimensions) {
      throw new IllegalArgumentException(
          "Expected " + dimensions + " vector dimensions, but received " + vector.length);
    }
  }
}

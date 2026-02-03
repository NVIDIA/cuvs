/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsFormat.HNSW_INDEX_CODEC_NAME;
import static com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsFormat.HNSW_INDEX_EXT;
import static com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsFormat.HNSW_META_CODEC_EXT;
import static com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsFormat.HNSW_META_CODEC_NAME;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.closeCuVSResourcesInstance;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.getCuVSResourcesInstance;
import static org.apache.lucene.index.VectorEncoding.FLOAT32;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.apache.lucene.util.RamUsageEstimator.shallowSizeOfInstance;

import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.RowView;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.logging.Logger;
import java.util.stream.IntStream;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.Sorter.DocMap;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.internal.hppc.IntObjectHashMap;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.HnswGraph.NodesIterator;
import org.apache.lucene.util.hnsw.NeighborArray;
import org.apache.lucene.util.packed.DirectMonotonicWriter;

/**
 * This class extends upon the KnnVectorsWriter to
 * enable the creation of GPU-based accelerated vector search indexes.
 *
 * @since 25.10
 */
public class Lucene99AcceleratedHNSWVectorsWriter extends KnnVectorsWriter {

  private static final long SHALLOW_RAM_BYTES_USED =
      shallowSizeOfInstance(Lucene99AcceleratedHNSWVectorsWriter.class);

  @SuppressWarnings("unused")
  private static final Logger log =
      Logger.getLogger(Lucene99AcceleratedHNSWVectorsWriter.class.getName());

  private static final String CUVS_COMPONENT = "CUVS";
  private static final LuceneProvider LUCENE_PROVIDER;
  private static final Integer VERSION_CURRENT;
  private static final List<VectorSimilarityFunction> VECTOR_SIMILARITY_FUNCTIONS;

  private final FlatVectorsWriter flatVectorsWriter; // for writing the raw vectors
  private final List<GPUFieldWriter> fields = new ArrayList<>();
  private final InfoStream infoStream;
  private IndexOutput cuvsIndex = null;
  private IndexOutput hnswMeta = null, hnswVectorIndex = null;
  private boolean finished;
  private String vemFileName;
  private String vexFileName;
  private final AcceleratedHNSWParams acceleratedHNSWParams;

  static {
    try {
      LUCENE_PROVIDER = LuceneProvider.getInstance("99");
      VERSION_CURRENT = LUCENE_PROVIDER.getStaticIntParam("VERSION_CURRENT");
      VECTOR_SIMILARITY_FUNCTIONS = LUCENE_PROVIDER.getSimilarityFunctions();
    } catch (Exception e) {
      throw new ExceptionInInitializerError(e.getMessage());
    }
  }

  /**
   * Initializes {@link Lucene99AcceleratedHNSWVectorsWriter}
   *
   * @param state instance of the {@link org.apache.lucene.index.SegmentWriteState}
   * @param acceleratedHNSWParams An instance of {@link AcceleratedHNSWParams}
   * @param flatVectorsWriter instance of the {@link org.apache.lucene.codecs.hnsw.FlatVectorsWriter}
   * @throws IOException IOException
   */
  public Lucene99AcceleratedHNSWVectorsWriter(
      SegmentWriteState state,
      AcceleratedHNSWParams acceleratedHNSWParams,
      FlatVectorsWriter flatVectorsWriter)
      throws IOException {
    super();
    this.flatVectorsWriter = flatVectorsWriter;
    this.infoStream = state.infoStream;
    this.acceleratedHNSWParams = acceleratedHNSWParams;

    vemFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, HNSW_META_CODEC_EXT);

    vexFileName =
        IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, HNSW_INDEX_EXT);

    boolean success = false;
    try {

      hnswMeta = state.directory.createOutput(vemFileName, state.context);
      hnswVectorIndex = state.directory.createOutput(vexFileName, state.context);

      CodecUtil.writeIndexHeader(
          hnswMeta,
          HNSW_META_CODEC_NAME,
          VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      CodecUtil.writeIndexHeader(
          hnswVectorIndex,
          HNSW_INDEX_CODEC_NAME,
          VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);

      success = true;
    } finally {
      if (success == false) {
        IOUtils.closeWhileHandlingException(this);
      }
    }
  }

  /**
   * Add new field for indexing.
   */
  @Override
  public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
    var encoding = fieldInfo.getVectorEncoding();
    if (encoding != FLOAT32) {
      throw new IllegalArgumentException("expected float32, got:" + encoding);
    }
    var writer = Objects.requireNonNull(flatVectorsWriter.addField(fieldInfo));
    @SuppressWarnings("unchecked")
    var flatWriter = (FlatFieldVectorsWriter<float[]>) writer;
    var cuvsFieldWriter = new GPUFieldWriter(fieldInfo, flatWriter);
    fields.add(cuvsFieldWriter);
    return writer;
  }

  /**
   * Builds an instance of CagraIndexParams.
   *
   * @return instance of CagraIndexParams
   */
  private CagraIndexParams cagraIndexParams() {
    // TODO: Make build algorithm configurable after fixing the related issue.
    return new CagraIndexParams.Builder()
        .withNumWriterThreads(acceleratedHNSWParams.getWriterThreads())
        .withIntermediateGraphDegree(acceleratedHNSWParams.getIntermediateGraphDegree())
        .withGraphDegree(acceleratedHNSWParams.getGraphdegree())
        .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
        .build();
  }

  /**
   * A utility method to print info/debugging messages using InfoStream.
   *
   * @param msg the debugging message to print
   */
  private void info(String msg) {
    if (infoStream.isEnabled(CUVS_COMPONENT)) {
      infoStream.message(CUVS_COMPONENT, msg);
    }
  }

  /**
   * Builds the intermediate CAGRA index and builds and writes the HNSW index
   *
   * @param fieldInfo instance of FieldInfo that has the field description
   * @param vectors vectors to index
   * @throws IOException
   */
  private void writeFieldInternal(FieldInfo fieldInfo, List<float[]> vectors) throws IOException {

    if (vectors.size() == 0) {
      writeEmpty(fieldInfo);
      return;
    }

    try {
      CuVSMatrix dataset =
          Utils.createFloatMatrix(
              vectors, fieldInfo.getVectorDimension(), getCuVSResourcesInstance());

      if (dataset.size() < 2) {
        // Handle single vector case by creating a dummy HNSW graph
        // TODO: Remove this workaround once https://github.com/rapidsai/cuvs/pull/1256 is merged
        writeSingleVectorGraph(fieldInfo, vectors);
        return;
      }

      long startTime = System.nanoTime();
      CagraIndexParams params = cagraIndexParams();
      CagraIndex cagraIndex =
          CagraIndex.newBuilder(getCuVSResourcesInstance())
              .withDataset(dataset)
              .withIndexParams(params)
              .build();

      // Get the adjacency list from CAGRA index
      CuVSMatrix adjacencyListMatrix = cagraIndex.getGraph();

      int size = (int) dataset.size();
      int dimensions = fieldInfo.getVectorDimension();

      // Create multi-layer HNSW graph from CAGRA
      GPUBuiltHnswGraph hnswGraph =
          createMultiLayerHnswGraph(
              fieldInfo,
              size,
              dimensions,
              adjacencyListMatrix,
              vectors,
              acceleratedHNSWParams.getHnswLayers());

      long vectorIndexOffset = hnswVectorIndex.getFilePointer();

      // Write the graph to the vector index
      int[][] graphLevelNodeOffsets = writeGraph(hnswGraph, hnswVectorIndex);

      long vectorIndexLength = hnswVectorIndex.getFilePointer() - vectorIndexOffset;

      // Write metadata
      writeMeta(
          hnswVectorIndex,
          hnswMeta,
          fieldInfo,
          vectorIndexOffset,
          vectorIndexLength,
          size,
          hnswGraph,
          graphLevelNodeOffsets);

      long elapsedMillis = Utils.nanosToMillis(System.nanoTime() - startTime);
      info("HNSW graph created in " + elapsedMillis + "ms, with " + dataset.size() + " vectors");

      cagraIndex.close();

    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
  }

  /**
   * Creates a multi-layer HNSW graph with dynamic number of layers.
   * M = cagraGraphDegree/2
   * Each layer contains 1/M nodes from the previous layer
   * Creates layers until the highest layer has ≤ M nodes
   */
  private GPUBuiltHnswGraph createMultiLayerHnswGraph(
      FieldInfo fieldInfo,
      int size,
      int dimensions,
      CuVSMatrix adjacencyListMatrix,
      List<float[]> vectors,
      int hnswLayers)
      throws Throwable {

    // Calculate M as cagraGraphDegree/2
    int M = acceleratedHNSWParams.getGraphdegree() / 2;

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
      float[][] selectedVectors = new float[nextLayerSize][];
      for (int i = 0; i < nextLayerSize; i++) {
        selectedVectors[i] = vectors.get(selectedNodes[i]);
      }

      // Build CAGRA graph for this layer
      layerAdjacencies.add(buildCagraGraphForSubset(selectedVectors, selectedNodes));

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
   * Builds a CAGRA graph for a subset of vectors
   */
  private CuVSMatrix buildCagraGraphForSubset(float[][] vectors, int[] selectedNodes)
      throws Throwable {
    // Create CuVSMatrix from the subset vectors
    CuVSMatrix subsetDataset = CuVSMatrix.ofArray(vectors);

    // Build CAGRA index for the subset
    CagraIndexParams params = cagraIndexParams();
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
  private void writeMeta(
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
    meta.writeVInt(acceleratedHNSWParams.getGraphdegree() / 2); // M = cagraGraphDegree/2

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

  /**
   * Returns a 2D array of offsets (information written while writing the meta info)
   *
   * @param graph instance of GPUBuiltHnswGraph
   * @param vectorIndex instance of IndexOutput
   * @return a 2D array of offsets
   * @throws IOException I/O Exceptions
   */
  private int[][] writeGraph(GPUBuiltHnswGraph graph, IndexOutput vectorIndex) throws IOException {
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
   * Build the indexes and writes it to the disk.
   */
  @Override
  public void flush(int maxDoc, DocMap sortMap) throws IOException {
    flatVectorsWriter.flush(maxDoc, sortMap);
    for (var field : fields) {
      if (sortMap == null) {
        writeField(field);
      } else {
        writeSortingField(field, sortMap);
      }
    }
  }

  /**
   * Builds the index and writes it to the disk.
   *
   * @param fieldData
   * @throws IOException
   */
  private void writeField(GPUFieldWriter fieldData) throws IOException {
    writeFieldInternal(fieldData.fieldInfo(), fieldData.getVectors());
  }

  /**
   * Builds the index and writes it to the disk.
   *
   * @param fieldData instance of GPUFieldWriter
   * @param sortMap instance of the DocMap
   * @throws IOException
   */
  private void writeSortingField(GPUFieldWriter fieldData, Sorter.DocMap sortMap)
      throws IOException {

    DocsWithFieldSet oldDocsWithFieldSet = fieldData.getDocsWithFieldSet();
    final int[] new2OldOrd = new int[oldDocsWithFieldSet.cardinality()]; // new ord to old ord
    mapOldOrdToNewOrd(oldDocsWithFieldSet, sortMap, null, new2OldOrd, null);

    List<float[]> sortedVectors = new ArrayList<float[]>();
    for (int i = 0; i < fieldData.getVectors().size(); i++) {
      sortedVectors.add(fieldData.getVectors().get(new2OldOrd[i]));
    }

    writeFieldInternal(fieldData.fieldInfo(), sortedVectors);
  }

  /**
   * Builds and writes a single vector graph.
   *
   * @param fieldInfo instance of FieldInfo
   * @param vectors the list of float vectors
   * @throws IOException I/O Exceptions
   */
  private void writeSingleVectorGraph(FieldInfo fieldInfo, List<float[]> vectors)
      throws IOException {
    // Workaround for CAGRA not supporting single vector indexes
    // TODO: Remove this method once https://github.com/rapidsai/cuvs/pull/1256 is merged
    try {
      int size = 1;
      int dimensions = fieldInfo.getVectorDimension();

      // Create a dummy HNSW graph for a single vector
      GPUBuiltHnswGraph hnswGraph = createSingleVectorHnswGraph(size, dimensions);

      long vectorIndexOffset = hnswVectorIndex.getFilePointer();

      // Write the graph to the vector index
      int[][] graphLevelNodeOffsets = writeGraph(hnswGraph, hnswVectorIndex);

      long vectorIndexLength = hnswVectorIndex.getFilePointer() - vectorIndexOffset;

      // Write metadata
      writeMeta(
          hnswVectorIndex,
          hnswMeta,
          fieldInfo,
          vectorIndexOffset,
          vectorIndexLength,
          size,
          hnswGraph,
          graphLevelNodeOffsets);

      long elapsedMillis = Utils.nanosToMillis(System.nanoTime() - System.nanoTime());
      info("Single vector HNSW graph created in " + elapsedMillis + "ms, with " + size + " vector");

    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
  }

  /**
   * Creates a dummy HNSW graph for a single vector.
   * The graph will have 1 level with 1 node and no neighbors.
   */
  private GPUBuiltHnswGraph createSingleVectorHnswGraph(int size, int dimensions) throws Throwable {
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
   * Writes an empty meta information for the field.
   *
   * @param fieldInfo instance of FieldInfo
   * @throws IOException I/O Exceptions
   */
  private void writeEmpty(FieldInfo fieldInfo) throws IOException {
    writeMeta(null, hnswMeta, fieldInfo, 0, 0, 0, null, null);
  }

  static int distFuncToOrd(VectorSimilarityFunction func) {
    for (int i = 0; i < VECTOR_SIMILARITY_FUNCTIONS.size(); i++) {
      if (VECTOR_SIMILARITY_FUNCTIONS.get(i).equals(func)) {
        return (byte) i;
      }
    }
    throw new IllegalArgumentException("invalid distance function: " + func);
  }

  /**
   * Uses the CAGRA API to merge the CAGRA indexes.
   *
   * @param fieldInfo instance of FieldInfo
   * @param mergeState instance of MergeState
   * @throws IOException I/O Exceptions
   */
  private void mergeCagraIndexes(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    try {

      List<CagraIndex> cagraIndexes = new ArrayList<>();
      // We need this count so that the merged segment's meta information has the vector count.
      int totalVectorCount = 0;

      for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
        KnnVectorsReader knnReader = mergeState.knnVectorsReaders[i];
        // Access the CAGRA index for this field from the reader

        if (knnReader != null) {
          if (knnReader instanceof CuVS2510GPUVectorsReader cvr) {
            if (cvr != null) {
              totalVectorCount += cvr.getFieldEntries().get(fieldInfo.number).count();
              CagraIndex cagraIndex = getCagraIndexFromReader(cvr, fieldInfo.name);
              if (cagraIndex != null) {
                cagraIndexes.add(cagraIndex);
              }
            }
          } else {
            // This should never happen
            throw new RuntimeException(
                "Reader is not of CuVSVectorsReader type. Instead it is: " + knnReader.getClass());
          }
        }
      }
      assert cagraIndexes.size() > 1;

      CagraIndex mergedIndex =
          CagraIndex.merge(cagraIndexes.toArray(new CagraIndex[cagraIndexes.size()]));
      writeMergedCagraIndex(fieldInfo, mergedIndex, totalVectorCount);
      info("Successfully merged " + cagraIndexes.size() + " CAGRA indexes using native merge API");

    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
  }

  /**
   * Creates List<Float[]> from merged vectors
   */
  private List<float[]> createListFromMergedVectors(FloatVectorValues mergedVectorValues)
      throws IOException {
    List<float[]> vectors = new ArrayList<float[]>();
    KnnVectorValues.DocIndexIterator iter = mergedVectorValues.iterator();
    for (int docV = iter.nextDoc(); docV != NO_MORE_DOCS; docV = iter.nextDoc()) {
      float[] vector = mergedVectorValues.vectorValue(iter.index());
      vectors.add(vector.clone());
    }
    return vectors;
  }

  /**
   * Fallback method that rebuilds indexes from merged vectors.
   * Used when native CAGRA merge() is not possible. Also used
   * when non-CAGRA index types are used (for e.g. Brute Force index).
   */
  private void vectorBasedMerge(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    if (fieldInfo.getVectorEncoding() != FLOAT32) {
      throw new AssertionError("Only Float32 supported");
    }
    try {
      List<float[]> dataset =
          createListFromMergedVectors(
              KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState));
      writeFieldInternal(fieldInfo, dataset);
    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
  }

  /**
   * Extracts the CAGRA index for a specific field from a CuVSVectorsReader.
   */
  private CagraIndex getCagraIndexFromReader(CuVS2510GPUVectorsReader reader, String fieldName) {
    try {
      IntObjectHashMap<GPUIndex> cuvsIndices = reader.getCuvsIndexes();
      FieldInfos fieldInfos = reader.getFieldInfos();
      FieldInfo fieldInfo = fieldInfos.fieldInfo(fieldName);

      if (fieldInfo != null) {
        GPUIndex cuvsIndex = cuvsIndices.get(fieldInfo.number);
        if (cuvsIndex != null) {
          return cuvsIndex.getCagraIndex();
        }
      }
    } catch (Exception e) {
      e.printStackTrace();
      info("Failed to extract CAGRA index for field " + fieldName + ": " + e.getMessage());
    }
    return null;
  }

  /**
   * Writes a pre-built merged CAGRA index to the output.
   */
  private void writeMergedCagraIndex(FieldInfo fieldInfo, CagraIndex mergedIndex, int vectorCount)
      throws IOException {
    try {
      var cagraIndexOutputStream = new IndexOutputOutputStream(cuvsIndex);

      // Serialize the merged index
      Path tmpFile =
          Files.createTempFile(getCuVSResourcesInstance().tempDirectory(), "mergedindex", "cag");
      mergedIndex.serialize(cagraIndexOutputStream, tmpFile);

      // TODO: Path to writeFieldInternal missing. Fix this.

      // Clean up the merged index
      mergedIndex.close();
    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
  }

  /**
   * Write field for merging.
   */
  @Override
  public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    flatVectorsWriter.mergeOneField(fieldInfo, mergeState);

    // Since CAGRA merge does not support merging of indexes with purging of deletes,
    // we fallback to vector-based re-indexing. Issue:
    // https://github.com/rapidsai/cuvs/issues/1253
    boolean hasDeletions =
        IntStream.range(0, mergeState.liveDocs.length)
            .anyMatch(
                i ->
                    mergeState.liveDocs[i] == null
                        || IntStream.range(0, mergeState.maxDocs[i])
                            .anyMatch(j -> !mergeState.liveDocs[i].get(j)));

    if (mergeState.knnVectorsReaders.length > 1 && !hasDeletions) {
      mergeCagraIndexes(fieldInfo, mergeState);
    } else {
      // CAGRA's merge API does not handle the trivial case of merging 1 index.
      vectorBasedMerge(fieldInfo, mergeState);
    }
  }

  /**
   * Called once at the end before close.
   */
  @Override
  public void finish() throws IOException {
    if (finished) {
      throw new IllegalStateException("already finished");
    }
    finished = true;
    flatVectorsWriter.finish();

    if (cuvsIndex != null) {
      CodecUtil.writeFooter(cuvsIndex);
    }

    if (hnswMeta != null) {
      // write end of fields marker
      hnswMeta.writeInt(-1);
      CodecUtil.writeFooter(hnswMeta);
    }
    if (hnswVectorIndex != null) {
      CodecUtil.writeFooter(hnswVectorIndex);
    }
  }

  /**
   * Closes the resources.
   */
  @Override
  public void close() throws IOException {
    IOUtils.close(cuvsIndex, hnswMeta, hnswVectorIndex, flatVectorsWriter);
    closeCuVSResourcesInstance();
  }

  /**
   * Returns the memory usage of this object in bytes.
   */
  @Override
  public long ramBytesUsed() {
    long total = SHALLOW_RAM_BYTES_USED;
    for (var field : fields) {
      total += field.ramBytesUsed();
    }
    return total;
  }
}

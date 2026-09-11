/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.CuVSTestSupport.enableRmmOrSkip;
import static com.nvidia.cuvs.lucene.CuVSTestSupport.requireCuvsOrSkip;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressFileSystems;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.HnswGraph.NodesIterator;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

/** Required GPU proof for the accelerated upper-layer build and persisted graph. */
@SuppressFileSystems("*")
@SuppressSysoutChecks(bugUrl = "")
public class TestAcceleratedThreeLayerRoundTrip extends LuceneTestCase {

  private static final int ROWS = 512;
  private static final int DIMENSIONS = 32;
  private static final int GRAPH_DEGREE = 8;
  private static final long LAYER_SEED = 0x2594L;
  private static final String VECTOR_FIELD = "vector";

  @BeforeClass
  public static void beforeClass() {
    enableRmmOrSkip();
  }

  @Before
  public void requireCuvs() {
    requireCuvsOrSkip();
  }

  @Test
  public void testAcceleratedThreeLayerGraphPersistsAndSearches() throws Exception {
    Path root = createTempDir("accelerated-three-layer");
    float[][] vectors = createVectors();
    Path fbin = writeFbin(root.resolve("vectors.fbin"), vectors);
    Path index = root.resolve("index");
    CagraHnswBuildMetrics metrics = new CagraHnswBuildMetrics();
    AcceleratedHNSWParams graphBuild =
        new AcceleratedHNSWParams.Builder()
            .withStrategy(AcceleratedHNSWParams.Strategy.CUSTOM)
            .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
            .withGraphDegree(GRAPH_DEGREE)
            .withIntermediateGraphDegree(16)
            .withHNSWLayer(3)
            .withHnswLayerSeed(LAYER_SEED)
            .build();
    CagraHnswBulkIndexWriter.Config config =
        CagraHnswBulkIndexWriter.Config.builder()
            .field(VECTOR_FIELD, DIMENSIONS, EUCLIDEAN)
            .graphBuild(graphBuild)
            .segments(1, false)
            .targetDirectory(index)
            .metrics(metrics)
            .build();

    CagraHnswBulkIndexWriter.indexMappedFbin(fbin, config);
    Files.delete(fbin);

    try (Directory directory = FSDirectory.open(index);
        DirectoryReader reader = DirectoryReader.open(directory)) {
      assertEquals(1, reader.leaves().size());
      LeafReader leaf = reader.leaves().getFirst().reader();
      HnswGraph graph = graphOf(leaf);
      assertEquals(3, graph.numLevels());
      List<int[]> expectedLayers =
          AcceleratedHNSWUtils.selectUpperLayerNodes(
              ROWS, 3, Math.ceilDiv(GRAPH_DEGREE, 2), LAYER_SEED);
      assertEquals(2, expectedLayers.size());
      for (int level = 1; level < graph.numLevels(); level++) {
        int[] nodes = NodesIterator.getSortedNodes(graph.getNodesOnLevel(level));
        assertArrayEquals(expectedLayers.get(level - 1), nodes);
        assertNeighborsRemainOnLevel(graph, level, nodes);
      }
      int[] topLevel = NodesIterator.getSortedNodes(graph.getNodesOnLevel(2));
      assertTrue(Arrays.binarySearch(topLevel, graph.entryNode()) >= 0);

      FloatVectorValues values = leaf.getFloatVectorValues(VECTOR_FIELD);
      assertArrayEquals(vectors[0], values.vectorValue(0), 0.0f);
      assertArrayEquals(vectors[ROWS - 1], values.copy().vectorValue(ROWS - 1), 0.0f);
      TopDocs hits =
          new IndexSearcher(reader).search(new KnnFloatVectorQuery(VECTOR_FIELD, vectors[0], 5), 5);
      assertEquals(5, hits.scoreDocs.length);
    }

    Map<String, Number> snapshot = metrics.snapshot();
    assertEquals(1L, snapshot.get("stage/base cagra-build [GPU]/count").longValue());
    assertEquals(1L, snapshot.get("stage/hnsw-convert [GPU+PCIe+CPU]/count").longValue());
    assertEquals(GRAPH_DEGREE, snapshot.get("gauge/effective graph degree").intValue());
  }

  @Test
  public void testMetricsSeparateSelectedAndNativeTruncatedGraphDegree() throws Exception {
    int rows = 20;
    int dimensions = 8;
    int selectedDegree = 32;
    Path root = createTempDir("actual-graph-degree");
    Path fbin = writeFbin(root.resolve("vectors.fbin"), createVectors(rows, dimensions));
    CagraHnswBuildMetrics metrics = new CagraHnswBuildMetrics();
    AcceleratedHNSWParams graphBuild =
        new AcceleratedHNSWParams.Builder()
            .withStrategy(AcceleratedHNSWParams.Strategy.CUSTOM)
            .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
            .withGraphDegree(selectedDegree)
            .withIntermediateGraphDegree(64)
            .build();
    CagraHnswBulkIndexWriter.Config config =
        CagraHnswBulkIndexWriter.Config.builder()
            .field(VECTOR_FIELD, dimensions, EUCLIDEAN)
            .graphBuild(graphBuild)
            .targetDirectory(root.resolve("index"))
            .metrics(metrics)
            .build();

    CagraHnswBulkIndexWriter.indexMappedFbin(fbin, config);

    Map<String, Number> snapshot = metrics.snapshot();
    int effectiveDegree = snapshot.get("gauge/effective graph degree").intValue();
    assertEquals(selectedDegree, snapshot.get("gauge/selected graph degree").intValue());
    assertEquals(rows - 1, effectiveDegree);
    assertEquals(
        (long) rows * effectiveDegree * Integer.BYTES,
        snapshot.get("counter/actual cagra adjacency bytes").longValue());
  }

  private static void assertNeighborsRemainOnLevel(HnswGraph graph, int level, int[] nodes)
      throws Exception {
    for (int node : nodes) {
      graph.seek(level, node);
      for (int neighbor = graph.nextNeighbor();
          neighbor != NO_MORE_DOCS;
          neighbor = graph.nextNeighbor()) {
        assertTrue(
            "upper-layer neighbor " + neighbor + " is absent from level " + level,
            Arrays.binarySearch(nodes, neighbor) >= 0);
      }
    }
  }

  private static HnswGraph graphOf(LeafReader leaf) throws Exception {
    KnnVectorsReader knnReader = ((CodecReader) leaf).getVectorReader();
    if (knnReader instanceof PerFieldKnnVectorsFormat.FieldsReader fieldsReader) {
      knnReader = fieldsReader.getFieldReader(VECTOR_FIELD);
    }
    return ((HnswGraphProvider) knnReader).getGraph(VECTOR_FIELD);
  }

  private static float[][] createVectors() {
    return createVectors(ROWS, DIMENSIONS);
  }

  private static float[][] createVectors(int rows, int dimensions) {
    float[][] vectors = new float[rows][dimensions];
    for (int row = 0; row < rows; row++) {
      for (int dimension = 0; dimension < dimensions; dimension++) {
        vectors[row][dimension] = (float) Math.sin((row + 1L) * (dimension + 3L) * 0.0009765625d);
      }
    }
    return vectors;
  }

  private static Path writeFbin(Path file, float[][] vectors) throws Exception {
    int rows = vectors.length;
    int dimensions = vectors[0].length;
    ByteBuffer data =
        ByteBuffer.allocate(2 * Integer.BYTES + rows * dimensions * Float.BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
    data.putInt(rows);
    data.putInt(dimensions);
    for (float[] vector : vectors) {
      for (float value : vector) {
        data.putFloat(value);
      }
    }
    data.flip();
    try (FileChannel channel =
        FileChannel.open(file, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
      while (data.hasRemaining()) {
        channel.write(data);
      }
    }
    return file;
  }
}

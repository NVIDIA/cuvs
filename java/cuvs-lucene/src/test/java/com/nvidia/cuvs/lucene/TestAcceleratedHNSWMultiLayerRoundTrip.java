/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import com.carrotsearch.randomizedtesting.annotations.Name;
import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.hnsw.HnswGraph;

/** Exercises the native upper-layer path through serialization and CPU search. */
@SuppressSysoutChecks(bugUrl = "")
public class TestAcceleratedHNSWMultiLayerRoundTrip extends LuceneTestCase {

  private static final String FIELD = "vector";
  private static final int VECTOR_COUNT = 256;
  private static final int GRAPH_DEGREE = 16;

  private final KnnVectorsFormat format;
  private final int dimensions;

  public TestAcceleratedHNSWMultiLayerRoundTrip(
      @Name("knnVectorsFormat") KnnVectorsFormat format, @Name("dimensions") int dimensions) {
    this.format = format;
    this.dimensions = dimensions;
  }

  @ParametersFactory
  public static List<Object[]> parameters() {
    AcceleratedHNSWParams params =
        new AcceleratedHNSWParams.Builder()
            .withStrategy(AcceleratedHNSWParams.Strategy.CUSTOM)
            .withIntermediateGraphDegree(32)
            .withGraphDegree(GRAPH_DEGREE)
            .withHNSWLayer(3)
            .build();
    return Arrays.asList(
        new Object[][] {
          {new Lucene99AcceleratedHNSWVectorsFormat(params), 32},
          {new LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat(params), 129},
          {new LuceneAcceleratedHNSWScalarQuantizedVectorsFormat(params), 32}
        });
  }

  public void testThreeLevelGraphCanBeReopenedAndSearched() throws Exception {
    assumeTrue("cuVS not supported", isSupported());
    float[][] vectors = randomVectors(VECTOR_COUNT, dimensions);

    try (Directory directory = newDirectory()) {
      IndexWriterConfig config =
          newIndexWriterConfig().setCodec(TestUtil.alwaysKnnVectorsFormat(format));
      try (IndexWriter writer = new IndexWriter(directory, config)) {
        for (int id = 0; id < vectors.length; id++) {
          Document document = new Document();
          document.add(new StringField("id", Integer.toString(id), Field.Store.YES));
          document.add(new KnnFloatVectorField(FIELD, vectors[id], EUCLIDEAN));
          writer.addDocument(document);
        }
        writer.forceMerge(1);
      }

      TestUtil.checkIndex(directory);
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        LeafReader leaf = getOnlyLeafReader(reader);
        HnswGraph graph = graphOf(leaf);
        assertEquals(3, graph.numLevels());

        List<Set<Integer>> nodesByLevel = collectNodes(graph);
        int expectedLevelOneSize = Math.max(2, VECTOR_COUNT / graph.maxConn());
        int expectedLevelTwoSize = Math.max(2, expectedLevelOneSize / graph.maxConn());
        assertEquals(VECTOR_COUNT, nodesByLevel.get(0).size());
        assertEquals(expectedLevelOneSize, nodesByLevel.get(1).size());
        assertEquals(expectedLevelTwoSize, nodesByLevel.get(2).size());
        assertTrue(nodesByLevel.get(0).containsAll(nodesByLevel.get(1)));
        assertTrue(nodesByLevel.get(1).containsAll(nodesByLevel.get(2)));
        assertUpperNeighborsStayOnTheirLevel(graph, nodesByLevel);

        int queryNode = graph.entryNode();
        assertTrue(nodesByLevel.get(2).contains(queryNode));
        IndexSearcher searcher = new IndexSearcher(reader);
        var hits = searcher.search(new KnnFloatVectorQuery(FIELD, vectors[queryNode], 10), 10);
        assertEquals(10, hits.scoreDocs.length);
        String queryNodeId = Integer.toString(queryNode);
        boolean foundQueryNode = false;
        for (var hit : hits.scoreDocs) {
          foundQueryNode |= queryNodeId.equals(searcher.storedFields().document(hit.doc).get("id"));
        }
        assertTrue("the entry-node vector must be returned for its own query", foundQueryNode);
      }
    }
  }

  private static List<Set<Integer>> collectNodes(HnswGraph graph) throws Exception {
    List<Set<Integer>> nodesByLevel = new ArrayList<>();
    for (int level = 0; level < graph.numLevels(); level++) {
      Set<Integer> nodes = new HashSet<>();
      HnswGraph.NodesIterator iterator = graph.getNodesOnLevel(level);
      while (iterator.hasNext()) {
        assertTrue("duplicate node on level " + level, nodes.add(iterator.nextInt()));
      }
      nodesByLevel.add(nodes);
    }
    return nodesByLevel;
  }

  private static void assertUpperNeighborsStayOnTheirLevel(
      HnswGraph graph, List<Set<Integer>> nodesByLevel) throws Exception {
    for (int level = 1; level < graph.numLevels(); level++) {
      Set<Integer> nodes = nodesByLevel.get(level);
      int arcCount = 0;
      for (int node : nodes) {
        graph.seek(level, node);
        for (int neighbor = graph.nextNeighbor();
            neighbor != NO_MORE_DOCS;
            neighbor = graph.nextNeighbor()) {
          assertTrue("negative upper-layer neighbor", neighbor >= 0);
          assertTrue("upper-layer neighbor outside the index", neighbor < VECTOR_COUNT);
          assertTrue(
              "upper-layer neighbor is not a member of level " + level, nodes.contains(neighbor));
          arcCount++;
        }
      }
      assertTrue("upper layer " + level + " contains no arcs", arcCount > 0);
    }
  }

  private static HnswGraph graphOf(LeafReader leaf) throws Exception {
    KnnVectorsReader reader = ((CodecReader) leaf).getVectorReader();
    if (reader instanceof PerFieldKnnVectorsFormat.FieldsReader fieldsReader) {
      reader = fieldsReader.getFieldReader(FIELD);
    }
    return ((HnswGraphProvider) reader).getGraph(FIELD);
  }

  private static float[][] randomVectors(int count, int dimensions) {
    Random random = new Random(0x2476L);
    float[][] vectors = new float[count][dimensions];
    for (float[] vector : vectors) {
      for (int dimension = 0; dimension < vector.length; dimension++) {
        vector[dimension] = random.nextFloat();
      }
    }
    return vectors;
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.generateDataset;
import static com.nvidia.cuvs.lucene.TestUtils.generateRandomVector;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.analysis.MockAnalyzer;
import org.apache.lucene.tests.analysis.MockTokenizer;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.junit.BeforeClass;
import org.junit.Test;

@SuppressSysoutChecks(bugUrl = "")
public class TestAcceleratedHNSWDeletedDocuments extends LuceneTestCase {

  protected static Logger log =
      Logger.getLogger(TestAcceleratedHNSWDeletedDocuments.class.getName());

  static final Codec codec =
      TestUtil.alwaysKnnVectorsFormat(new Lucene99AcceleratedHNSWVectorsFormat());
  private static Random random;

  @BeforeClass
  public static void beforeClass() throws Exception {
    assumeTrue("cuVS not supported", isSupported());
    random = random();
  }

  @Test
  public void testVectorSearchWithDeletedDocuments() throws IOException {

    try (Directory directory = newDirectory()) {
      int datasetSize = random.nextInt(200, 1000); // 200-1200 documents
      int dimensions = random.nextInt(64, 256); // 64-320 dimensions
      int topK = Math.min(random.nextInt(20) + 5, datasetSize / 2); // 5-25 results
      float deletionProbability = random.nextFloat() * 0.4f + 0.1f; // 10-50% deletion rate

      float[][] dataset = generateDataset(random, datasetSize, dimensions);
      Set<Integer> deletedDocs = new HashSet<>();

      // Create index with all documents having vectors
      try (RandomIndexWriter writer = createWriter(directory)) {
        for (int i = 0; i < datasetSize; i++) {
          Document doc = new Document();
          doc.add(new StringField("id", String.valueOf(i), Field.Store.YES));
          doc.add(
              new KnnFloatVectorField("vector", dataset[i], VectorSimilarityFunction.EUCLIDEAN));
          writer.addDocument(doc);
        }

        // Delete documents randomly based on probability
        for (int i = 0; i < datasetSize; i++) {
          if (random.nextFloat() < deletionProbability) {
            writer.deleteDocuments(new Term("id", String.valueOf(i)));
            deletedDocs.add(i);
          }
        }
        writer.commit();
      }

      // Search and verify deleted documents are not returned
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        IndexSearcher searcher = newSearcher(reader);
        // Use a random vector for query
        float[] queryVector = generateRandomVector(dimensions, random);

        Query query = new KnnFloatVectorQuery("vector", queryVector, topK);
        ScoreDoc[] hits = searcher.search(query, topK).scoreDocs;

        // Verify we got results
        assertTrue("Should have search results", hits.length > 0);

        // Verify no deleted documents in results
        for (ScoreDoc hit : hits) {
          String docId = reader.storedFields().document(hit.doc).get("id");
          int id = Integer.parseInt(docId);
          assertFalse(
              "Deleted document " + id + " should not appear in results", deletedDocs.contains(id));
          log.log(Level.FINE, "Found non-deleted document: " + id + ", Score: " + hit.score);
        }

        // Verify deleted documents are truly deleted
        for (int deletedId : deletedDocs) {
          TopDocs result =
              searcher.search(new TermQuery(new Term("id", String.valueOf(deletedId))), 1);
          assertEquals(
              "Deleted document " + deletedId + " should not be found",
              0,
              result.totalHits.value());
        }
      }
    }
  }

  @Test
  public void testVectorSearchWithMixedDeletedAndMissingVectors() throws IOException {

    try (Directory directory = newDirectory()) {
      int datasetSize = random.nextInt(200) + 50; // 50-250 documents
      int dimensions = random.nextInt(256) + 64; // 64-320 dimensions
      int topK = Math.min(random.nextInt(20) + 5, datasetSize / 2); // 5-25 results
      float vectorProbability = random.nextFloat() * 0.5f + 0.3f; // 30-80% have vectors
      float deletionProbability = random.nextFloat() * 0.3f + 0.1f; // 10-40% deletion rate

      float[][] dataset = generateDataset(random, datasetSize, dimensions);
      Set<Integer> docsWithoutVectors = new HashSet<>();
      Set<Integer> deletedDocs = new HashSet<>();

      // Create index with mixed documents
      try (RandomIndexWriter writer = createWriter(directory)) {
        for (int i = 0; i < datasetSize; i++) {
          Document doc = new Document();
          doc.add(new StringField("id", String.valueOf(i), Field.Store.YES));
          // Randomly assign categories
          String category = random.nextBoolean() ? "A" : "B";
          doc.add(new StringField("category", category, Field.Store.YES));

          // Randomly decide whether to add vectors
          if (random.nextFloat() < vectorProbability) {
            doc.add(
                new KnnFloatVectorField("vector", dataset[i], VectorSimilarityFunction.EUCLIDEAN));
          } else {
            docsWithoutVectors.add(i);
          }
          writer.addDocument(doc);
        }

        // Delete documents randomly
        for (int i = 0; i < datasetSize; i++) {
          if (random.nextFloat() < deletionProbability) {
            writer.deleteDocuments(new Term("id", String.valueOf(i)));
            deletedDocs.add(i);
          }
        }
        writer.commit();
      }

      // Test vector search behavior
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        IndexSearcher searcher = newSearcher(reader);
        float[] queryVector = generateRandomVector(dimensions, random);

        Query query = new KnnFloatVectorQuery("vector", queryVector, topK);
        ScoreDoc[] hits = searcher.search(query, topK).scoreDocs;

        // Verify results
        for (ScoreDoc hit : hits) {
          String docId = reader.storedFields().document(hit.doc).get("id");
          int id = Integer.parseInt(docId);
          assertFalse("Deleted document should not appear", deletedDocs.contains(id));
          assertFalse("Document without vector should not appear", docsWithoutVectors.contains(id));
          log.log(Level.FINE, "Found document with vector: " + id + ", Score: " + hit.score);
        }

        // Test filtered search with deletions
        Query filter = new TermQuery(new Term("category", "A"));
        Query filteredQuery = new KnnFloatVectorQuery("vector", queryVector, topK, filter);
        ScoreDoc[] filteredHits = searcher.search(filteredQuery, topK).scoreDocs;

        for (ScoreDoc hit : filteredHits) {
          Document doc = reader.storedFields().document(hit.doc);
          String category = doc.get("category");
          assertEquals("Should only match category A", "A", category);
          int id = Integer.parseInt(doc.get("id"));
          assertFalse(
              "Deleted document should not appear in filtered results", deletedDocs.contains(id));
        }
      }
    }
  }

  @Test
  public void testVectorSearchAfterAllDocumentsDeleted() throws IOException {

    try (Directory directory = newDirectory()) {
      int datasetSize = random.nextInt(20) + 5; // 5-25 documents for this test
      int dimensions = random.nextInt(128) + 32; // 32-160 dimensions
      int topK = Math.min(random.nextInt(10) + 5, datasetSize); // 5-15 results

      float[][] dataset = generateDataset(random, datasetSize, dimensions);

      // Create and delete all documents
      try (IndexWriter writer = new IndexWriter(directory, createWriterConfig())) {
        for (int i = 0; i < datasetSize; i++) {
          Document doc = new Document();
          doc.add(new StringField("id", String.valueOf(i), Field.Store.YES));
          doc.add(
              new KnnFloatVectorField("vector", dataset[i], VectorSimilarityFunction.EUCLIDEAN));
          writer.addDocument(doc);
        }
        writer.commit();

        // Delete all documents
        for (int i = 0; i < datasetSize; i++) {
          writer.deleteDocuments(new Term("id", String.valueOf(i)));
        }
        writer.commit();
        writer.forceMerge(1); // Force merge to apply deletions
      }

      // Verify search returns no results
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        IndexSearcher searcher = newSearcher(reader);
        float[] queryVector = generateRandomVector(dimensions, random);

        Query query = new KnnFloatVectorQuery("vector", queryVector, topK);
        TopDocs results = searcher.search(query, topK);

        assertEquals(
            "Should return no results when all documents are deleted",
            0,
            results.totalHits.value());
      }
    }
  }

  @Test
  public void testVectorSearchWithPartialDeletionAndReindexing() throws IOException {

    try (Directory directory = newDirectory()) {
      int datasetSize = random.nextInt(200) + 50; // 50-250 documents
      int dimensions = random.nextInt(256) + 64; // 64-320 dimensions
      int topK = Math.min(random.nextInt(20) + 5, datasetSize / 2); // 5-25 results
      float deletionProbability = random.nextFloat() * 0.3f + 0.1f; // 10-40% deletion rate

      float[][] dataset = generateDataset(random, datasetSize, dimensions);
      List<Integer> activeDocIds = new ArrayList<>();

      // Initial indexing
      try (IndexWriter writer = new IndexWriter(directory, createWriterConfig())) {
        int initialDocs = datasetSize / 2 + random.nextInt(datasetSize / 4); // 50-75% of dataset
        for (int i = 0; i < initialDocs; i++) {
          Document doc = new Document();
          doc.add(new StringField("id", String.valueOf(i), Field.Store.YES));
          doc.add(
              new KnnFloatVectorField("vector", dataset[i], VectorSimilarityFunction.EUCLIDEAN));
          writer.addDocument(doc);
          activeDocIds.add(i);
        }

        // Delete some documents randomly
        List<Integer> candidatesForDeletion = new ArrayList<>(activeDocIds);
        for (int docId : candidatesForDeletion) {
          if (random.nextFloat() < deletionProbability) {
            writer.deleteDocuments(new Term("id", String.valueOf(docId)));
            activeDocIds.remove(Integer.valueOf(docId));
          }
        }

        // Add new documents with higher IDs
        for (int i = initialDocs; i < datasetSize; i++) {
          Document doc = new Document();
          doc.add(new StringField("id", String.valueOf(i), Field.Store.YES));
          doc.add(
              new KnnFloatVectorField("vector", dataset[i], VectorSimilarityFunction.EUCLIDEAN));
          writer.addDocument(doc);
          activeDocIds.add(i);
        }
        writer.commit();
      }

      // Verify search behavior after deletions and additions
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        IndexSearcher searcher = newSearcher(reader);
        float[] queryVector = generateRandomVector(dimensions, random);

        Query query = new KnnFloatVectorQuery("vector", queryVector, topK);
        ScoreDoc[] hits = searcher.search(query, topK).scoreDocs;

        Set<Integer> resultIds = new HashSet<>();
        for (ScoreDoc hit : hits) {
          String docId = reader.storedFields().document(hit.doc).get("id");
          int id = Integer.parseInt(docId);
          resultIds.add(id);
          assertTrue("Result should be from active documents", activeDocIds.contains(id));
        }

        log.log(
            Level.FINE,
            "Search returned "
                + hits.length
                + " results from "
                + activeDocIds.size()
                + " active documents");
      }
    }
  }

  @Test
  public void testForceMergeCountsOnlyLiveSparseVectors() throws Exception {
    final String vectorField = "vector";
    final int dimensions = 129;
    Map<String, float[]> expected = new LinkedHashMap<>();

    try (Directory directory = newDirectory()) {
      try (IndexWriter writer =
          new IndexWriter(directory, createWriterConfig().setMergePolicy(NoMergePolicy.INSTANCE))) {
        for (int segment = 0; segment < 3; segment++) {
          for (int row = 0; row < 5; row++) {
            String id = segment + "-" + row;
            Document document = new Document();
            document.add(new StringField("id", id, Field.Store.YES));
            if (row < 4) {
              float[] vector = deterministicVector(segment * 5 + row, dimensions);
              document.add(
                  new KnnFloatVectorField(vectorField, vector, VectorSimilarityFunction.EUCLIDEAN));
              if (row != 1) {
                expected.put(id, vector);
              }
            }
            writer.addDocument(document);
          }
          writer.commit();
        }
        for (int segment = 0; segment < 3; segment++) {
          writer.deleteDocuments(new Term("id", segment + "-1"));
        }
        writer.commit();

        try (DirectoryReader sourceReader = DirectoryReader.open(writer)) {
          assertEquals("the test requires three source segments", 3, sourceReader.leaves().size());
          for (var context : sourceReader.leaves()) {
            LeafReader sourceLeaf = context.reader();
            assertTrue("each source segment must carry a deletion", sourceLeaf.hasDeletions());
            assertEquals(5, sourceLeaf.maxDoc());
            assertEquals(4, sourceLeaf.numDocs());
            assertEquals(4, sourceLeaf.getFloatVectorValues(vectorField).size());
          }
        }

        writer.getConfig().setMergePolicy(new TieredMergePolicy());
        writer.forceMerge(1);
      }

      TestUtil.checkIndex(directory);
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        var leaf = getOnlyLeafReader(reader);
        var values = leaf.getFloatVectorValues(vectorField);
        assertNotNull(values);
        assertEquals(expected.size(), values.size());
        Set<String> seen = new HashSet<>();
        for (int ordinal = 0; ordinal < values.size(); ordinal++) {
          String id = leaf.storedFields().document(values.ordToDoc(ordinal)).get("id");
          assertTrue("Unexpected or duplicate vector for " + id, seen.add(id));
          assertNotNull("No expected vector for " + id, expected.get(id));
          assertArrayEquals(expected.get(id), values.vectorValue(ordinal), 0.0f);
        }
        assertEquals(expected.keySet(), seen);

        HnswGraph graph = graphOf(leaf, vectorField);
        assertEquals(values.size(), graph.size());
        assertEquals(values.size(), graph.getNodesOnLevel(0).size());
        assertAllGraphOrdinalsInBounds(graph, values.size());
      }
    }
  }

  @Test
  public void testForceMergeWithExactlyOneLiveVector() throws Exception {
    assertTrivialLiveVectorMerge(1);
  }

  @Test
  public void testForceMergeWithZeroLiveVectors() throws Exception {
    assertTrivialLiveVectorMerge(0);
  }

  private void assertTrivialLiveVectorMerge(int liveVectors) throws Exception {
    final String vectorField = "vector";
    final int dimensions = 129;

    try (Directory directory = newDirectory()) {
      try (IndexWriter writer =
          new IndexWriter(directory, createWriterConfig().setMergePolicy(NoMergePolicy.INSTANCE))) {
        for (int id = 0; id < 3; id++) {
          Document vectorDocument = new Document();
          vectorDocument.add(new StringField("id", "vector-" + id, Field.Store.YES));
          vectorDocument.add(
              new KnnFloatVectorField(
                  vectorField,
                  deterministicVector(id, dimensions),
                  VectorSimilarityFunction.EUCLIDEAN));
          writer.addDocument(vectorDocument);

          Document sparseDocument = new Document();
          sparseDocument.add(new StringField("id", "sparse-" + id, Field.Store.YES));
          writer.addDocument(sparseDocument);
          writer.commit();
        }
        for (int id = liveVectors; id < 3; id++) {
          writer.deleteDocuments(new Term("id", "vector-" + id));
        }
        writer.commit();

        try (DirectoryReader sourceReader = DirectoryReader.open(writer)) {
          assertEquals("the test requires three source segments", 3, sourceReader.leaves().size());
        }

        writer.getConfig().setMergePolicy(new TieredMergePolicy());
        writer.forceMerge(1);
      }

      TestUtil.checkIndex(directory);
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        LeafReader leaf = getOnlyLeafReader(reader);
        FloatVectorValues values = leaf.getFloatVectorValues(vectorField);
        assertNotNull(values);
        assertEquals(liveVectors, values.size());

        HnswGraph graph = graphOf(leaf, vectorField);
        assertEquals(liveVectors, graph.size());
        assertEquals(liveVectors == 0 ? 0 : 1, graph.numLevels());
        assertEquals(liveVectors, graph.getNodesOnLevel(0).size());
        assertAllGraphOrdinalsInBounds(graph, liveVectors);
        if (liveVectors == 1) {
          graph.seek(0, 0);
          assertEquals(
              "a single-node graph must have no edges", NO_MORE_DOCS, graph.nextNeighbor());
        }

        ((CodecReader) leaf).getVectorReader().checkIntegrity();
        IndexSearcher searcher = new IndexSearcher(reader);
        TopDocs results =
            searcher.search(
                new KnnFloatVectorQuery(vectorField, deterministicVector(0, dimensions), 1), 1);
        assertEquals(liveVectors, results.totalHits.value());
        assertEquals(liveVectors, results.scoreDocs.length);
        if (liveVectors == 1) {
          assertEquals(
              "vector-0", searcher.storedFields().document(results.scoreDocs[0].doc).get("id"));
        }
      }
    }
  }

  private static HnswGraph graphOf(LeafReader leaf, String field) throws Exception {
    KnnVectorsReader reader = ((CodecReader) leaf).getVectorReader();
    if (reader instanceof PerFieldKnnVectorsFormat.FieldsReader fieldsReader) {
      reader = fieldsReader.getFieldReader(field);
    }
    return ((HnswGraphProvider) reader).getGraph(field);
  }

  private static void assertAllGraphOrdinalsInBounds(HnswGraph graph, int vectorCount)
      throws Exception {
    for (int level = 0; level < graph.numLevels(); level++) {
      HnswGraph.NodesIterator nodes = graph.getNodesOnLevel(level);
      while (nodes.hasNext()) {
        int node = nodes.nextInt();
        assertTrue("graph node is outside the vector domain", node >= 0 && node < vectorCount);
        graph.seek(level, node);
        for (int neighbor = graph.nextNeighbor();
            neighbor != NO_MORE_DOCS;
            neighbor = graph.nextNeighbor()) {
          assertTrue(
              "graph neighbor is outside the vector domain",
              neighbor >= 0 && neighbor < vectorCount);
        }
      }
    }
  }

  private static float[] deterministicVector(int id, int dimensions) {
    float[] vector = new float[dimensions];
    for (int dimension = 0; dimension < dimensions; dimension++) {
      vector[dimension] = id * 10.0f + dimension * 0.01f;
    }
    return vector;
  }

  private RandomIndexWriter createWriter(Directory directory) throws IOException {
    return new RandomIndexWriter(
        random(),
        directory,
        newIndexWriterConfig(new MockAnalyzer(random(), MockTokenizer.SIMPLE, true))
            .setCodec(codec)
            .setMergePolicy(newTieredMergePolicy()));
  }

  private IndexWriterConfig createWriterConfig() {
    return newIndexWriterConfig(new MockAnalyzer(random(), MockTokenizer.SIMPLE, true))
        .setCodec(codec)
        .setMergePolicy(newTieredMergePolicy());
  }
}

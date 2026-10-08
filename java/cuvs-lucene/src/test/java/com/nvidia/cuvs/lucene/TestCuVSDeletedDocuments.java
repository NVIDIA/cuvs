/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.generateDataset;
import static com.nvidia.cuvs.lucene.TestUtils.generateRandomVector;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.analysis.MockAnalyzer;
import org.apache.lucene.tests.analysis.MockTokenizer;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.Bits;
import org.junit.BeforeClass;
import org.junit.Test;

@SuppressSysoutChecks(bugUrl = "")
public class TestCuVSDeletedDocuments extends LuceneTestCase {

  protected static Logger log = Logger.getLogger(TestCuVSDeletedDocuments.class.getName());

  static final Codec codec = TestUtil.alwaysKnnVectorsFormat(new CuVS2510GPUVectorsFormat());
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

        GPUKnnFloatVectorQuery query =
            new GPUKnnFloatVectorQuery("vector", queryVector, topK, null, topK, 1);
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

        GPUKnnFloatVectorQuery query =
            new GPUKnnFloatVectorQuery("vector", queryVector, topK, null, topK, 1);
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
        GPUKnnFloatVectorQuery filteredQuery =
            new GPUKnnFloatVectorQuery("vector", queryVector, topK, filter, topK, 1);
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

        GPUKnnFloatVectorQuery query =
            new GPUKnnFloatVectorQuery("vector", queryVector, topK, null, topK, 1);
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

        GPUKnnFloatVectorQuery query =
            new GPUKnnFloatVectorQuery("vector", queryVector, topK, null, topK, 1);
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

  /**
   * A segment whose vector-bearing documents are all deleted must contribute zero hits rather than
   * failing. Such a segment still reaches the reader as long as it keeps at least one live document
   * without a vector, so Lucene does not drop it. See
   * <a href="https://github.com/NVIDIA/cuvs/issues/2599">issue 2599</a>: the accepted-ordinal set is
   * empty, which used to clamp the cuVS top-k to zero and yield a result list with no rows at all.
   */
  @Test
  public void testSearchSegmentWithAllVectorsDeleted() throws IOException {

    final int dimensions = 64;
    final int liveDocs = 16;
    final int topK = 5;

    try (Directory directory = newDirectory()) {
      float[][] dataset = generateDataset(random, liveDocs + 1, dimensions);

      // NoMergePolicy keeps the two commits as two separate segments, so the first one survives as
      // a leaf with a single, deleted vector.
      try (IndexWriter writer =
          new IndexWriter(directory, createWriterConfig().setMergePolicy(NoMergePolicy.INSTANCE))) {

        // Segment 1: one document with a vector (deleted below) and one live document without a
        // vector, which is what keeps the segment from being dropped once the first is deleted.
        Document withVector = new Document();
        withVector.add(new StringField("id", "deleted-vector", Field.Store.YES));
        withVector.add(
            new KnnFloatVectorField("vector", dataset[0], VectorSimilarityFunction.EUCLIDEAN));
        writer.addDocument(withVector);

        Document withoutVector = new Document();
        withoutVector.add(new StringField("id", "no-vector", Field.Store.YES));
        writer.addDocument(withoutVector);
        writer.commit();

        // Segment 2: live vectors, so the query still has something to return.
        for (int i = 0; i < liveDocs; i++) {
          Document doc = new Document();
          doc.add(new StringField("id", "live-" + i, Field.Store.YES));
          doc.add(
              new KnnFloatVectorField(
                  "vector", dataset[i + 1], VectorSimilarityFunction.EUCLIDEAN));
          writer.addDocument(doc);
        }
        writer.commit();

        writer.deleteDocuments(new Term("id", "deleted-vector"));
        writer.commit();
      }

      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        assertTrue("Expected more than one segment", reader.leaves().size() > 1);
        IndexSearcher searcher = new IndexSearcher(reader);
        float[] queryVector = generateRandomVector(dimensions, random);

        // KnnFloatVectorQuery always goes through the per-segment reader path, which is where the
        // empty accepted-ordinal set is handled.
        assertOnlyLiveHits(
            reader, searcher.search(new KnnFloatVectorQuery("vector", queryVector, topK), topK));

        // GPUKnnFloatVectorQuery may take either the multi-partition or the per-segment path
        // depending on whether every segment has a usable CAGRA index; both must behave the same.
        assertOnlyLiveHits(
            reader,
            searcher.search(
                new GPUKnnFloatVectorQuery("vector", queryVector, topK, null, topK, 1), topK));
      }
    }
  }

  /**
   * The reader must treat any empty accepted-ordinal set as "no hits", whatever produced it.
   *
   * <p>Going through {@link KnnFloatVectorQuery} with an explicit filter cannot reach this state:
   * Lucene ANDs every user filter with a {@code FieldExistsQuery} on the vector field, so the
   * accepted set always holds at least one vector-bearing document. This test therefore drives
   * {@link org.apache.lucene.index.LeafReader#searchNearestVectors} directly with an all-false
   * {@link Bits}, pinning the reader's own contract independently of the query layer.
   */
  @Test
  public void testSearchWithAcceptDocsMatchingNoVectors() throws IOException {

    final int dimensions = 64;
    final int vectorDocs = 16;
    final int topK = 5;

    try (Directory directory = newDirectory()) {
      float[][] dataset = generateDataset(random, vectorDocs, dimensions);

      try (IndexWriter writer = new IndexWriter(directory, createWriterConfig())) {
        for (int i = 0; i < vectorDocs; i++) {
          Document doc = new Document();
          doc.add(new StringField("id", String.valueOf(i), Field.Store.YES));
          doc.add(
              new KnnFloatVectorField("vector", dataset[i], VectorSimilarityFunction.EUCLIDEAN));
          writer.addDocument(doc);
        }
        writer.commit();
      }

      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        float[] queryVector = generateRandomVector(dimensions, random);
        for (LeafReaderContext ctx : reader.leaves()) {
          TopKnnCollector collector = new TopKnnCollector(topK, Integer.MAX_VALUE);
          ctx.reader()
              .searchNearestVectors(
                  "vector", queryVector, collector, new Bits.MatchNoBits(ctx.reader().maxDoc()));
          assertEquals(
              "An empty accepted-ordinal set should collect nothing",
              0,
              collector.topDocs().scoreDocs.length);
        }
      }
    }
  }

  /** Asserts every hit comes from a live, vector-bearing document. */
  private void assertOnlyLiveHits(DirectoryReader reader, TopDocs topDocs) throws IOException {
    assertTrue("Expected hits from the live segment", topDocs.scoreDocs.length > 0);
    for (ScoreDoc hit : topDocs.scoreDocs) {
      String id = reader.storedFields().document(hit.doc).get("id");
      assertTrue("Unexpected hit: " + id, id.startsWith("live-"));
    }
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

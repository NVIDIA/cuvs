/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.generateDataset;
import static com.nvidia.cuvs.lucene.TestUtils.generateRandomVector;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.logging.Logger;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
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
import org.junit.BeforeClass;
import org.junit.Test;

@SuppressSysoutChecks(bugUrl = "prints info from within cuvs")
public class TestCuVSDeletedDocuments extends LuceneTestCase {

  protected static Logger log = Logger.getLogger(TestCuVSDeletedDocuments.class.getName());

  static final Codec codec = TestUtil.alwaysKnnVectorsFormat(new CuVSVectorsFormat());
  private static Random random;

  @BeforeClass
  public static void beforeClass() throws Exception {
    assumeTrue("cuvs not supported", CuVSVectorsFormat.supported());
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
          log.info("Found non-deleted document: " + id + ", Score: " + hit.score);
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
          log.info("Found document with vector: " + id + ", Score: " + hit.score);
        }

        // Test filtered search with deletions
        Query filter = new TermQuery(new Term("category", "A"));
        Query filteredQuery =
            new CuVSKnnFloatVectorQuery("vector", queryVector, topK, filter, topK, 1);
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

        log.info(
            "Search returned "
                + hits.length
                + " results from "
                + activeDocIds.size()
                + " active documents");
      }
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

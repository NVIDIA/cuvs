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

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.logging.Logger;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.analysis.MockAnalyzer;
import org.apache.lucene.tests.analysis.MockTokenizer;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.apache.lucene.tests.util.English;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.tests.util.TestUtil;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

@SuppressSysoutChecks(bugUrl = "prints info from within cuvs")
public class TestCuVSGaps extends LuceneTestCase {

  protected static Logger log = Logger.getLogger(TestCuVSGaps.class.getName());

  static final Codec codec = TestUtil.alwaysKnnVectorsFormat(new CuVSVectorsFormat());
  static IndexSearcher searcher;
  static IndexReader reader;
  static Directory directory;
  static Random random;

  static int DATASET_SIZE_LIMIT = 1000;
  static int DIMENSIONS_LIMIT = 2048;
  static int NUM_QUERIES_LIMIT = 10;
  static int TOP_K_LIMIT = 64;

  static int datasetSize;
  static int dimension;
  static float[][] dataset;

  @BeforeClass
  public static void beforeClass() throws Exception {
    assertTrue("cuvs not supported", CuVSVectorsFormat.supported());
    directory = newDirectory();
    random = random();

    RandomIndexWriter writer =
        new RandomIndexWriter(
            random(),
            directory,
            newIndexWriterConfig(new MockAnalyzer(random(), MockTokenizer.SIMPLE, true))
                .setMaxBufferedDocs(TestUtil.nextInt(random(), 100, 1000))
                .setCodec(codec)
                .setMergePolicy(newTieredMergePolicy()));

    log.info("Merge Policy: " + writer.w.getConfig().getMergePolicy());

    datasetSize = random.nextInt(100, DATASET_SIZE_LIMIT);
    dimension = random.nextInt(8, DIMENSIONS_LIMIT);
    dataset = generateDataset(random, datasetSize, dimension);

    // Create documents where only even-numbered documents have vectors
    for (int i = 0; i < datasetSize; i++) {
      Document doc = new Document();
      doc.add(new StringField("id", String.valueOf(i), Field.Store.YES));
      doc.add(newTextField("field", English.intToEnglish(i), Field.Store.YES));

      // Only add vectors to even-numbered documents
      if (i % 2 == 0) {
        doc.add(new KnnFloatVectorField("vector", dataset[i], VectorSimilarityFunction.EUCLIDEAN));
      }

      writer.addDocument(doc);
    }

    reader = writer.getReader();
    searcher = newSearcher(reader);
    writer.close();
  }

  @AfterClass
  public static void afterClass() throws Exception {
    if (reader != null) reader.close();
    if (directory != null) directory.close();
    searcher = null;
    reader = null;
    directory = null;
    log.info("Test finished");
  }

  @Test
  public void testVectorSearchWithAlternatingDocuments() throws IOException {
    assertTrue("cuvs not supported", CuVSVectorsFormat.supported());

    // Use the first vector (from document 0) as query
    float[] queryVector = dataset[0];
    int topK = random.nextInt(5, TOP_K_LIMIT);

    Query query = new KnnFloatVectorQuery("vector", queryVector, topK);
    ScoreDoc[] hits = searcher.search(query, topK).scoreDocs;

    // Verify we get exactly TOP_K results
    assertEquals("Should return exactly " + topK + " results", topK, hits.length);

    // Verify all returned documents have vectors (even-numbered IDs)
    for (ScoreDoc hit : hits) {
      String docId = reader.storedFields().document(hit.doc).get("id");
      int id = Integer.parseInt(docId);
      assertEquals("All results should be even-numbered (have vectors)", 0, id % 2);
      log.info("Document ID: " + id + ", Score: " + hit.score);
    }

    // Verify the results match expected top-k based on Euclidean distance
    List<Integer> expectedIds = calculateExpectedTopK(queryVector, topK, dataset);
    for (int i = 0; i < hits.length; i++) {
      String docId = reader.storedFields().document(hits[i].doc).get("id");
      int id = Integer.parseInt(docId);
      assertTrue("Result " + id + " should be in expected top-k results", expectedIds.contains(id));
    }

    log.info("Alternating document test passed with " + hits.length + " results");
  }

  @Test
  public void testVectorSearchWithFilterAndAlternatingDocuments() throws IOException {
    assumeTrue("cuvs not supported", CuVSVectorsFormat.supported());

    // Use the first vector (from document 0) as query
    float[] queryVector = dataset[0];
    int topK = random.nextInt(5, TOP_K_LIMIT);

    // Create a filter that only matches documents with ID less than 10
    // This should further restrict our results to even numbers 0, 2, 4, 6, 8
    Query filter = new TermQuery(new Term("id", "8")); // Only match document 8

    Query filteredQuery = new CuVSKnnFloatVectorQuery("vector", queryVector, topK, filter, topK, 1);
    ScoreDoc[] filteredHits = searcher.search(filteredQuery, topK).scoreDocs;

    // Should only get document 8 (the only one that matches the filter and has a vector)
    assertEquals("Should return exactly 1 result", 1, filteredHits.length);

    String docId = reader.storedFields().document(filteredHits[0].doc).get("id");
    assertEquals("Should only return document 8", "8", docId);

    log.info("Filtered alternating document test passed with " + filteredHits.length + " results");
  }

  public static List<Integer> calculateExpectedTopK(float[] query, int topK, float[][] dataset) {
    Map<Integer, Double> distances = new TreeMap<>();

    // Calculate distances only for documents that have vectors (even-numbered)
    for (int i = 0; i < dataset.length; i += 2) {
      double distance = 0;
      for (int j = 0; j < dataset[0].length; j++) {
        distance += (query[j] - dataset[i][j]) * (query[j] - dataset[i][j]);
      }
      distances.put(i, distance);
    }

    // Sort by distance and return top-k
    return distances.entrySet().stream()
        .sorted(Map.Entry.comparingByValue())
        .map(Map.Entry::getKey)
        .limit(topK)
        .toList();
  }
}

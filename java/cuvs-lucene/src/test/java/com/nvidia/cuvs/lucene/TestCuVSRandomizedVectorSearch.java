/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.generateDataset;
import static com.nvidia.cuvs.lucene.TestUtils.generateQueries;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.logging.Level;
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

@SuppressSysoutChecks(bugUrl = "")
public class TestCuVSRandomizedVectorSearch extends LuceneTestCase {

  protected static Logger log = Logger.getLogger(TestCuVSRandomizedVectorSearch.class.getName());

  static final Codec codec = TestUtil.alwaysKnnVectorsFormat(new CuVS2510GPUVectorsFormat());
  static IndexSearcher searcher;
  static IndexReader reader;
  static Directory directory;

  static int DATASET_SIZE_LIMIT = 1000;
  static int DIMENSIONS_LIMIT = 2048;
  static int NUM_QUERIES_LIMIT = 10;
  static int TOP_K_LIMIT = 64; // TODO This fails beyond 64
  static float[][] dataset;

  @BeforeClass
  public static void beforeClass() throws Exception {
    assumeTrue("cuVS not supported", isSupported());
    directory = newDirectory();

    RandomIndexWriter writer =
        new RandomIndexWriter(
            random(),
            directory,
            newIndexWriterConfig(new MockAnalyzer(random(), MockTokenizer.SIMPLE, true))
                .setMaxBufferedDocs(TestUtil.nextInt(random(), 100, 1000))
                .setCodec(codec)
                .setMergePolicy(newTieredMergePolicy()));

    log.log(Level.FINE, "Merge Policy: " + writer.w.getConfig().getMergePolicy());

    Random random = random();
    int datasetSize = random.nextInt(DATASET_SIZE_LIMIT) + 1;
    int dimensions = random.nextInt(DIMENSIONS_LIMIT) + 1;
    dataset = generateDataset(random, datasetSize, dimensions);
    for (int i = 0; i < datasetSize; i++) {
      Document doc = new Document();
      doc.add(new StringField("id", String.valueOf(i), Field.Store.YES));
      doc.add(newTextField("field", English.intToEnglish(i), Field.Store.YES));
      boolean skipVector =
          random.nextInt(10)
              < 4; // some documents won't have vectors to test deleted/missing vectors
      if (!skipVector
          || datasetSize < 100) { // about 10th of the documents shouldn't have a single vector
        doc.add(new KnnFloatVectorField("vector", dataset[i], VectorSimilarityFunction.EUCLIDEAN));
        doc.add(new KnnFloatVectorField("vector2", dataset[i], VectorSimilarityFunction.EUCLIDEAN));
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
    log.log(Level.FINE, "Test finished");
  }

  @Test
  public void testVectorSearch() throws IOException {
    Random random = random();
    int numQueries = random.nextInt(NUM_QUERIES_LIMIT) + 1;
    int topK = Math.min(random.nextInt(TOP_K_LIMIT) + 1, dataset.length);

    if (dataset.length < topK) topK = dataset.length;

    float[][] queries = generateQueries(random, dataset[0].length, numQueries);
    List<List<Integer>> expected = generateExpectedResults(topK, dataset, queries);

    log.log(Level.FINE, "Dataset size: " + dataset.length + "x" + dataset[0].length);
    log.log(Level.FINE, "Query size: " + numQueries + "x" + queries[0].length);
    log.log(Level.FINE, "TopK: " + topK);

    GPUKnnFloatVectorQuery query =
        new GPUKnnFloatVectorQuery("vector", queries[0], topK, null, topK, 1);
    int correct[] = new int[topK];
    for (int i = 0; i < topK; i++) correct[i] = expected.get(0).get(i);

    ScoreDoc[] hits = searcher.search(query, topK).scoreDocs;
    log.log(Level.FINE, "RESULTS: " + Arrays.toString(hits));
    log.log(Level.FINE, "EXPECTD: " + expected.get(0));

    for (ScoreDoc hit : hits) {
      log.log(
          Level.FINE, "\t" + reader.storedFields().document(hit.doc).get("id") + ": " + hit.score);
    }

    for (ScoreDoc hit : hits) {
      int doc = Integer.parseInt(reader.storedFields().document(hit.doc).get("id"));
      assertTrue("Result returned was not in topk*2: " + doc, expected.get(0).contains(doc));
    }
  }

  private static List<List<Integer>> generateExpectedResults(
      int topK, float[][] dataset, float[][] queries) {
    List<List<Integer>> neighborsResult = new ArrayList<>();
    int dimensions = dataset[0].length;

    for (float[] query : queries) {
      Map<Integer, Double> distances = new TreeMap<>();
      for (int j = 0; j < dataset.length; j++) {
        double distance = 0;
        for (int k = 0; k < dimensions; k++) {
          distance += (query[k] - dataset[j][k]) * (query[k] - dataset[j][k]);
        }
        distances.put(j, (distance));
      }

      Map<Integer, Double> sorted = new TreeMap<Integer, Double>(distances);
      log.log(Level.FINE, "EXPECTED: " + sorted);

      // Sort by distance and select the topK nearest neighbors
      List<Integer> neighbors =
          distances.entrySet().stream()
              .sorted(Map.Entry.comparingByValue())
              .map(Map.Entry::getKey)
              .toList();
      neighborsResult.add(neighbors.subList(0, Math.min(topK * 3, dataset.length)));
    }

    log.log(Level.FINE, "Expected results generated successfully.");
    return neighborsResult;
  }

  @Test
  public void testVectorSearchWithFilter() throws IOException {
    assumeTrue("cuVS not supported", isSupported());

    Random random = random();
    int topK = Math.min(random.nextInt(TOP_K_LIMIT) + 1, dataset.length);

    if (dataset.length < topK) topK = dataset.length;

    // Find a document that has a vector by doing a search first
    GPUKnnFloatVectorQuery unfiltered =
        new GPUKnnFloatVectorQuery("vector", dataset[0], 1, null, 1, 1);
    ScoreDoc[] unfilteredHits = searcher.search(unfiltered, 1).scoreDocs;

    // Skip test if no vectors found at all
    assumeTrue(
        "Need at least one document with vector for filtering test", unfilteredHits.length > 0);

    String targetDocId = reader.storedFields().document(unfilteredHits[0].doc).get("id");
    float[] queryVector = dataset[0];

    // Create a filter that matches only the document we know has a vector
    Query filter = new TermQuery(new Term("id", targetDocId));

    // Test the new constructor with filter
    GPUKnnFloatVectorQuery filteredQuery =
        new GPUKnnFloatVectorQuery("vector", queryVector, topK, filter, topK, 1);

    ScoreDoc[] filteredHits = searcher.search(filteredQuery, topK).scoreDocs;

    // Ensure we got some results
    assertTrue("Should have at least one result", filteredHits.length > 0);

    // Verify that all results match the filter
    for (ScoreDoc hit : filteredHits) {
      String docId = reader.storedFields().document(hit.doc).get("id");
      assertEquals("All results should match the filter", targetDocId, docId);
    }

    log.log(Level.FINE, "Prefiltering test passed with " + filteredHits.length + " results");
  }
}

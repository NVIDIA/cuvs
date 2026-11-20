/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.assertSearchFindsNearestNeighbors;
import static com.nvidia.cuvs.lucene.TestUtils.generateDataset;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import com.carrotsearch.randomizedtesting.RandomSupplier;
import java.nio.file.Path;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.Before;
import org.junit.Test;

@SuppressSysoutChecks(bugUrl = "")
public class TestCagraToHnswSerializationAndSearch extends LuceneTestCase {

  // The degree sweep used this method seed with the test runner's default RNG.
  private static final long RECALL_SWEEP_TEST_SEED = 0xB76BE3D67B11916AL;

  private static Logger log =
      Logger.getLogger(TestCagraToHnswSerializationAndSearch.class.getName());

  @Before
  public void beforeTest() throws Exception {
    assumeTrue("cuVS not supported", isSupported());
  }

  @Test
  public void testCagraToHnswSerializationAndSearch() throws Exception {
    // At 0.50, at most one of the 50 results per run fell outside the brute-force top 15
    // in the fixed-seed sweep. Allow four misses on randomized inputs. Randomly wired graphs
    // in the original calibration produced at least five misses per run.
    assertRoundTripSearch(0.50, 4, random());
  }

  @Test
  public void testRecallWithGraphDegreeFraction06() throws Exception {
    // Replay the sweep's dataset and queries to check exact top-5 recall at higher fractions.
    // The fewest exact hits out of 50 results were 32 at 0.60 and 34 at 0.80 (below).
    // Allow two fewer exact hits for build variability and at most three results outside
    // their query's top 15.
    assertExactRecall(0.60, 30, RandomSupplier.DEFAULT.get(RECALL_SWEEP_TEST_SEED));
  }

  @Test
  public void testRecallWithGraphDegreeFraction08() throws Exception {
    assertExactRecall(0.80, 32, RandomSupplier.DEFAULT.get(RECALL_SWEEP_TEST_SEED));
  }

  private void assertExactRecall(double fraction, int minimumExactHits, Random vectorRandom)
      throws Exception {
    int exactHits = assertRoundTripSearch(fraction, 3, vectorRandom);
    assertTrue(
        "Expected at least "
            + minimumExactHits
            + " exact top-5 hits at graph degree fraction "
            + fraction
            + ", got "
            + exactHits,
        exactHits >= minimumExactHits);
  }

  private int assertRoundTripSearch(double fraction, int allowedMisses, Random vectorRandom)
      throws Exception {
    AcceleratedHNSWParams params =
        new AcceleratedHNSWParams.Builder().withVariableGraphDegreeFraction(fraction).build();
    Codec codec = CuVSCodecs.acceleratedHNSW(params);
    IndexWriterConfig config = new IndexWriterConfig().setCodec(codec).setUseCompoundFile(false);

    final int COMMIT_FREQ = 2000;
    final String ID_FIELD = "id";
    final String VECTOR_FIELD = "vector_field";

    int numDocs = 2000;
    int dimension = 32;
    int topK = 5;
    int numQueries = 10;
    int count = COMMIT_FREQ;
    float[][] dataset = generateDataset(vectorRandom, numDocs, dimension);
    Path indexDirPath = createTempDir();

    // Indexing
    try (Directory indexDirectory = FSDirectory.open(indexDirPath);
        IndexWriter indexWriter = new IndexWriter(indexDirectory, config)) {
      for (int i = 0; i < numDocs; i++) {
        Document document = new Document();
        document.add(new StringField(ID_FIELD, Integer.toString(i), Field.Store.YES));
        document.add(new KnnFloatVectorField(VECTOR_FIELD, dataset[i], EUCLIDEAN));
        indexWriter.addDocument(document);
        count -= 1;
        if (count == 0) {
          indexWriter.commit();
          count = COMMIT_FREQ;
        }
      }
    }

    // Searching
    try (Directory indexDirectory = FSDirectory.open(indexDirPath);
        DirectoryReader reader = DirectoryReader.open(indexDirectory)) {
      log.log(Level.FINE, "Successfully opened index");

      int vectorCount = 0;
      for (LeafReaderContext leafReaderContext : reader.leaves()) {
        LeafReader leafReader = leafReaderContext.reader();
        FloatVectorValues knnValues = leafReader.getFloatVectorValues(VECTOR_FIELD);
        assertNotNull(knnValues);
        log.log(
            Level.FINE,
            VECTOR_FIELD
                + " field: "
                + knnValues.size()
                + " vectors, "
                + knnValues.dimension()
                + " dimensions");
        vectorCount += knnValues.size();
        assertTrue("Vector dimension mismatch", knnValues.dimension() == dimension);
      }
      assertTrue("Dataset size mismatch", vectorCount == numDocs);

      log.log(Level.FINE, "Testing vector search queries...");
      IndexSearcher searcher = new IndexSearcher(reader);

      float[][] queries = generateDataset(vectorRandom, numQueries, dimension);
      return assertSearchFindsNearestNeighbors(
          searcher, VECTOR_FIELD, ID_FIELD, dataset, queries, topK, allowedMisses);
    }
  }

  @Test
  public void testSingleVectorIndex() throws Exception {
    // Test single vector index support with dummy HNSW graph
    // TODO: This test can be removed once https://github.com/NVIDIA/cuvs/pull/1256 is merged
    // and CAGRA natively supports single vector indexes
    Codec codec = CuVSCodecs.acceleratedHNSW();

    final String ID_FIELD = "id";
    final String VECTOR_FIELD = "vector_field";

    int dimension = 32;
    float[] vector = generateDataset(random(), 1, dimension)[0];

    // Index a single document with a vector - this should now work with dummy HNSW graph
    try (Directory indexDirectory = newDirectory()) {
      IndexWriterConfig config = new IndexWriterConfig().setCodec(codec).setUseCompoundFile(false);
      try (IndexWriter indexWriter = new IndexWriter(indexDirectory, config)) {
        Document document = new Document();
        document.add(new StringField(ID_FIELD, "0", Field.Store.YES));
        document.add(new KnnFloatVectorField(VECTOR_FIELD, vector, EUCLIDEAN));
        indexWriter.addDocument(document);

        // This should now succeed by creating a dummy HNSW graph for the single vector
        indexWriter.commit();
      }

      // Verify the index can be opened and searched
      try (DirectoryReader reader = DirectoryReader.open(indexDirectory)) {
        assertEquals(1, reader.numDocs());
        LeafReader leafReader = getOnlyLeafReader(reader);
        FloatVectorValues knnValues = leafReader.getFloatVectorValues(VECTOR_FIELD);
        assertNotNull(knnValues);
        assertEquals(1, knnValues.size());
        assertEquals(dimension, knnValues.dimension());

        // Test search functionality
        IndexSearcher searcher = new IndexSearcher(reader);
        KnnFloatVectorQuery query = new KnnFloatVectorQuery(VECTOR_FIELD, vector, 1);
        TopDocs results = searcher.search(query, 1);
        assertEquals(1, results.totalHits.value());
        assertEquals(1, results.scoreDocs.length);
        assertEquals(0, results.scoreDocs[0].doc);
      }
    }
  }
}

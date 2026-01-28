/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.generateDataset;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.UUID;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.commons.io.FileUtils;
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
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

@SuppressSysoutChecks(bugUrl = "")
public class TestCagraToHnswSerializationAndSearch extends LuceneTestCase {

  private static Logger log =
      Logger.getLogger(TestCagraToHnswSerializationAndSearch.class.getName());
  private static Random random;
  private static Path indexDirPath;

  @Before
  public void beforeTest() throws Exception {
    assumeTrue("cuVS not supported", isSupported());
    // Fixed seed so that we can validate against the same result.
    random = new Random(222);
    indexDirPath = Paths.get(UUID.randomUUID().toString());
  }

  @Test
  public void testCagraToHnswSerializationAndSearch() throws Exception {
    Codec codec = new Lucene101AcceleratedHNSWCodec(1, 128, 64, 3, 16, 100);
    IndexWriterConfig config = new IndexWriterConfig().setCodec(codec).setUseCompoundFile(false);

    final int COMMIT_FREQ = 2000;
    final String ID_FIELD = "id";
    final String VECTOR_FIELD = "vector_field";

    int numDocs = 2000;
    int dimension = 32;
    int topK = 5;
    int count = COMMIT_FREQ;
    float[][] dataset = generateDataset(random, numDocs, dimension);

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

      float[] queryVector = generateDataset(random, 1, dimension)[0];
      log.log(Level.FINE, "Query vector: " + Arrays.toString(queryVector));

      KnnFloatVectorQuery query = new KnnFloatVectorQuery(VECTOR_FIELD, queryVector, topK);
      TopDocs results = searcher.search(query, topK);

      log.log(Level.FINE, "Search results (" + results.totalHits + " total hits):");
      Integer[] expected = new Integer[] {1869, 1803, 1302, 59, 1497, 108, 1411, 351, 1982};
      HashSet<Integer> expectedIds = new HashSet<Integer>(Arrays.asList(expected));

      for (int i = 0; i < results.scoreDocs.length; i++) {
        ScoreDoc scoreDoc = results.scoreDocs[i];
        Document doc = searcher.storedFields().document(scoreDoc.doc);
        String id = doc.get(ID_FIELD);
        log.log(
            Level.FINE,
            "  Rank "
                + (i + 1)
                + ": doc "
                + scoreDoc.doc
                + " (id="
                + id
                + "), score="
                + scoreDoc.score);
        assertTrue(
            "Id: " + id + " expected but not found", expectedIds.contains(Integer.valueOf(id)));
      }
      assertTrue("TopK results not returned", results.scoreDocs.length == topK);
    }
  }

  @Test
  public void testSingleVectorIndex() throws Exception {
    // Test single vector index support with dummy HNSW graph
    // TODO: This test can be removed once https://github.com/rapidsai/cuvs/pull/1256 is merged
    // and CAGRA natively supports single vector indexes
    Codec codec = new Lucene101AcceleratedHNSWCodec();

    final String ID_FIELD = "id";
    final String VECTOR_FIELD = "vector_field";

    int dimension = 32;
    float[] vector = generateDataset(random, 1, dimension)[0];

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

  @After
  public void afterTest() throws Exception {
    File indexDirPathFile = indexDirPath.toFile();
    if (indexDirPathFile.exists() && indexDirPathFile.isDirectory()) {
      FileUtils.deleteDirectory(indexDirPathFile);
    }
  }
}

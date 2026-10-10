/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.assertSearchFindsNearestNeighbors;
import static com.nvidia.cuvs.lucene.TestUtils.generateDataset;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.closeCuVSResourcesInstance;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.setCuVSResourcesInstance;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import java.nio.file.Path;
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
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

@SuppressSysoutChecks(bugUrl = "")
public class TestCagraToHnswSerializationAndSearchWithFallbackWriter extends LuceneTestCase {

  private static Logger log =
      Logger.getLogger(TestCagraToHnswSerializationAndSearchWithFallbackWriter.class.getName());

  @Before
  public void beforeTest() throws Exception {
    assumeTrue("cuVS not supported", isSupported());
    // Simulate that cuVS is not supported. The resources are per thread, so this has to happen on
    // the thread that runs the test, and the handle being replaced has to be closed.
    closeCuVSResourcesInstance();
    setCuVSResourcesInstance(null);
  }

  @Test
  public void testCagraToHnswSerializationAndSearchWithFallbackWriter() throws Exception {
    assertFalse("cuVS resources should be disabled to force the fallback writer", isSupported());
    AcceleratedHNSWParams params =
        new AcceleratedHNSWParams.Builder()
            .withHNSWLayer(3)
            .withMaxConn(16)
            .withBeamWidth(100)
            .build();
    Codec codec = new Lucene101AcceleratedHNSWCodec(params);

    IndexWriterConfig config = new IndexWriterConfig().setCodec(codec).setUseCompoundFile(false);

    final int COMMIT_FREQ = 2000;
    final String ID_FIELD = "id";
    final String VECTOR_FIELD = "vector_field";

    int numDocs = 2000;
    int dimension = 32;
    int topK = 5;
    int numQueries = 10;
    int count = COMMIT_FREQ;
    float[][] dataset = generateDataset(random(), numDocs, dimension);
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
    try (Directory indexDirectory = FSDirectory.open(indexDirPath)) {
      try (DirectoryReader reader = DirectoryReader.open(indexDirectory)) {
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

        float[][] queries = generateDataset(random(), numQueries, dimension);
        // Lucene builds this graph, so this only checks that the fallback's index is searchable.
        // Out of the 50 results, its graph put up to 8 outside the closest 15 over 5500 random
        // datasets.
        int allowedMisses = 10;
        assertSearchFindsNearestNeighbors(
            searcher, VECTOR_FIELD, ID_FIELD, dataset, queries, topK, allowedMisses);
      }
    }
  }

  @After
  public void afterTest() throws Exception {
    // Drop the null so that the thread's next use creates working resources for other tests.
    closeCuVSResourcesInstance();
  }
}

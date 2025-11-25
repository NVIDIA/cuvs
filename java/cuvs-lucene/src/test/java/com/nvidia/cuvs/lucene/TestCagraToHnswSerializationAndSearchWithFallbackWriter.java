/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.generateDataset;
import static com.nvidia.cuvs.lucene.Utils.cuVSResourcesOrNull;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.UUID;
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
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

@SuppressSysoutChecks(bugUrl = "")
public class TestCagraToHnswSerializationAndSearchWithFallbackWriter extends LuceneTestCase {

  private static Logger log =
      Logger.getLogger(TestCagraToHnswSerializationAndSearchWithFallbackWriter.class.getName());

  private static Random random;
  private static Path indexDirPath;

  @BeforeClass
  public static void beforeClass() throws Exception {
    assumeTrue("cuVS not supported", Lucene99AcceleratedHNSWVectorsFormat.supported());
    // Set resources to null to simulate that cuVS is not supported.
    Lucene99AcceleratedHNSWVectorsFormat.setResources(null);
    // Fixed seed so that we can validate against the same result.
    random = new Random(222);
    indexDirPath = Paths.get(UUID.randomUUID().toString());
  }

  @Test
  public void testCagraToHnswSerializationAndSearchWithFallbackWriter()
      throws IOException,
          ClassNotFoundException,
          NoSuchMethodException,
          SecurityException,
          InstantiationException,
          IllegalAccessException,
          IllegalArgumentException,
          InvocationTargetException {
    Codec codec = new Lucene101AcceleratedHNSWCodec(32, 128, 64, 3, 16, 100);
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
    try (Directory indexDirectory = FSDirectory.open(indexDirPath)) {
      try (DirectoryReader reader = DirectoryReader.open(indexDirectory)) {
        log.info("Successfully opened index");

        int vectorCount = 0;
        for (LeafReaderContext leafReaderContext : reader.leaves()) {
          LeafReader leafReader = leafReaderContext.reader();
          FloatVectorValues knnValues = leafReader.getFloatVectorValues(VECTOR_FIELD);
          assertNotNull(knnValues);
          log.info(
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

        log.info("Testing vector search queries...");
        IndexSearcher searcher = new IndexSearcher(reader);

        float[] queryVector = generateDataset(random, 1, dimension)[0];
        log.info("Query vector: " + Arrays.toString(queryVector));

        KnnFloatVectorQuery query = new KnnFloatVectorQuery(VECTOR_FIELD, queryVector, topK);
        TopDocs results = searcher.search(query, topK);

        log.info("Search results (" + results.totalHits + " total hits):");
        Integer[] expected = new Integer[] {1869, 1411, 1497, 351, 554};
        HashSet<Integer> expectedIds = new HashSet<Integer>(Arrays.asList(expected));

        for (int i = 0; i < results.scoreDocs.length; i++) {
          ScoreDoc scoreDoc = results.scoreDocs[i];
          Document doc = searcher.storedFields().document(scoreDoc.doc);
          String id = doc.get(ID_FIELD);
          log.info(
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

      } catch (Exception e) {
        e.printStackTrace();
      }
    }
  }

  @AfterClass
  public static void afterClass() throws Exception {
    // Reset resources for other tests to work
    Lucene99AcceleratedHNSWVectorsFormat.setResources(cuVSResourcesOrNull());
    File indexDirPathFile = indexDirPath.toFile();
    if (indexDirPathFile.exists() && indexDirPathFile.isDirectory()) {
      FileUtils.deleteDirectory(indexDirPathFile);
    }
  }
}

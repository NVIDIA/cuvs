/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.generateDataset;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
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
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.English;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.tests.util.TestUtil;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

@SuppressSysoutChecks(bugUrl = "")
public class TestMultithreadedCuVSGPUSearch extends LuceneTestCase {

  private static final Logger log =
      Logger.getLogger(TestMultithreadedCuVSGPUSearch.class.getName());
  private static final Codec codec =
      TestUtil.alwaysKnnVectorsFormat(new CuVS2510GPUVectorsFormat());
  private static final String VECTOR_FIELD = "vectors";

  private static Directory directory;
  private static Random random;
  private static BlockingQueue<float[]> queries;
  private static int numQueries;
  private static int topK;
  private static int numThreads;

  @BeforeClass
  public static void beforeClass() throws IOException {
    assumeTrue("cuVS not supported", isSupported());
    random = random();
    directory = newDirectory(new ByteBuffersDirectory());
    IndexWriterConfig config = new IndexWriterConfig().setCodec(codec);
    IndexWriter writer = new IndexWriter(directory, config);

    int datasetSize = random.nextInt(500, 2000);
    int dimensions = random.nextInt(64, 256);
    topK = random.nextInt(2, 30);
    log.log(Level.FINE, "Using topK as: " + topK);
    numThreads = random.nextInt(2, 8);
    log.log(Level.FINE, "Generating a dataset with " + datasetSize + " vectors");
    float[][] dataset = generateDataset(random, datasetSize, dimensions);
    numQueries = random.nextInt(100, 500);
    log.log(Level.FINE, "Generating a query set with " + numQueries + " queries");
    float[][] queryVectors = generateDataset(random, numQueries, dimensions);
    queries = new ArrayBlockingQueue<>(numQueries, true, Arrays.asList(queryVectors));

    log.log(Level.FINE, "Indexing " + datasetSize + " vectors");
    for (int i = 0; i < datasetSize; i++) {
      Document doc = new Document();
      doc.add(new StringField("id", String.valueOf(i), Field.Store.YES));
      doc.add(newTextField("text_field", English.intToEnglish(i), Field.Store.YES));
      doc.add(new KnnFloatVectorField(VECTOR_FIELD, dataset[i], EUCLIDEAN));
      writer.addDocument(doc);
    }
    writer.commit();
    writer.close();
  }

  @Test
  public void testMultithreadedCuVSGPUSearch() throws Exception {
    DirectoryReader reader = DirectoryReader.open(directory);
    ExecutorService executorService = Executors.newFixedThreadPool(numThreads);
    IndexSearcher searcher = new IndexSearcher(reader);
    CountDownLatch latch = new CountDownLatch(numThreads);
    AtomicInteger totalSuccessfulQueries = new AtomicInteger();
    log.log(Level.FINE, "Using " + numThreads + " threads");
    for (int i = 0; i < numThreads; i++) {
      executorService.execute(
          new Runnable() {
            public void run() {
              try {
                float[] queryVector;
                String threadName = Thread.currentThread().getName();
                while ((queryVector = queries.poll()) != null) {
                  log.log(Level.FINE, "Thread: " + threadName + ", queue size: " + queries.size());
                  log.log(Level.FINER, "Query: " + Arrays.toString(queryVector));
                  GPUKnnFloatVectorQuery query =
                      new GPUKnnFloatVectorQuery(VECTOR_FIELD, queryVector, topK, null, topK, 1);
                  ScoreDoc[] hits = searcher.search(query, topK).scoreDocs;
                  totalSuccessfulQueries.addAndGet(hits.length == topK ? 1 : 0);
                }
              } catch (Exception e) {
                e.printStackTrace();
              } finally {
                latch.countDown();
              }
            }
          });
    }
    latch.await();
    executorService.shutdown();
    executorService.close();
    reader.close();
    log.log(
        Level.FINE,
        "Number queries that returned topK values: " + totalSuccessfulQueries.intValue());
    assertEquals(
        "All search queries did not return topK results",
        totalSuccessfulQueries.intValue(),
        numQueries);
  }

  @AfterClass
  public static void afterClass() throws IOException {
    if (directory != null) directory.close();
    directory = null;
    log.log(Level.FINE, "Test finished");
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.closeCuVSResourcesInstance;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.setCuVSResourcesInstance;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Handler;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.AfterClass;
import org.junit.BeforeClass;

/**
 * Checks that the accelerated HNSW codecs build the graph on the CPU when cuVS is not available, and
 * that they read back what their CPU fallback wrote.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestQuantizedFallbackWriters extends LuceneTestCase {

  private static final int NUM_DOCS = 300;
  private static final int DIMENSIONS = 16;
  private static final int TOP_K = 10;

  @BeforeClass
  public static void beforeClass() {
    // Where cuVS is unavailable (no GPU, JDK 21) the writers fall back by themselves. Elsewhere,
    // make isSupported() false on this thread, like on a host without a GPU, after closing the
    // resources isSupported() just created, so that they do not leak.
    if (isSupported()) {
      closeCuVSResourcesInstance();
      setCuVSResourcesInstance(null);
    }
  }

  @AfterClass
  public static void afterClass() {
    // Forget the null set above; the next use on this thread creates resources again, if any.
    closeCuVSResourcesInstance();
  }

  public void testFallbackWriters() throws Exception {
    AcceleratedHNSWParams params =
        new AcceleratedHNSWParams.Builder().withMaxConn(16).withBeamWidth(100).build();
    for (Codec codec :
        List.of(
            CuVSCodecs.acceleratedHNSW(params),
            CuVSCodecs.acceleratedHNSWScalarQuantized(params),
            CuVSCodecs.acceleratedHNSWBinaryQuantized(params))) {
      checkWritesAndReads(codec);
    }
  }

  private void checkWritesAndReads(Codec codec) throws Exception {
    Random random = new Random(random().nextLong());
    float[][] vectors = new float[NUM_DOCS][DIMENSIONS];
    AtomicInteger fallbacks = new AtomicInteger();
    Handler countFallbacks =
        new Handler() {
          @Override
          public void publish(LogRecord record) {
            if (record.getMessage() != null && record.getMessage().contains("falling back")) {
              fallbacks.incrementAndGet();
            }
          }

          @Override
          public void flush() {}

          @Override
          public void close() {}
        };
    Logger logger = Logger.getLogger("com.nvidia.cuvs.lucene");
    logger.addHandler(countFallbacks);
    try (Directory dir = newDirectory()) {
      // Flushes and merges on this thread, where cuVS is unavailable.
      IndexWriterConfig config =
          new IndexWriterConfig()
              .setCodec(codec)
              .setUseCompoundFile(false)
              .setMergeScheduler(new SerialMergeScheduler());
      try (IndexWriter writer = new IndexWriter(dir, config)) {
        for (int i = 0; i < NUM_DOCS; i++) {
          for (int d = 0; d < DIMENSIONS; d++) {
            vectors[i][d] = random.nextFloat();
          }
          Document doc = new Document();
          doc.add(new StoredField("id", i));
          doc.add(new KnnFloatVectorField("vector", vectors[i], EUCLIDEAN));
          writer.addDocument(doc);
        }
      }
      assertTrue(codec.getName() + " did not fall back to the CPU", fallbacks.get() > 0);
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        assertEquals(codec.getName(), NUM_DOCS, reader.numDocs());
        IndexSearcher searcher = new IndexSearcher(reader);
        int found = 0;
        for (int i = 0; i < NUM_DOCS; i++) {
          ScoreDoc[] hits =
              searcher.search(new KnnFloatVectorQuery("vector", vectors[i], TOP_K), TOP_K)
                  .scoreDocs;
          for (ScoreDoc hit : hits) {
            if (searcher.storedFields().document(hit.doc).getField("id").numericValue().intValue()
                == i) {
              found++;
              break;
            }
          }
        }
        assertTrue(
            codec.getName() + ": only " + found + " of " + NUM_DOCS + " vectors found themselves",
            found >= NUM_DOCS * 0.9);
      }
    } finally {
      logger.removeHandler(countFallbacks);
    }
  }
}

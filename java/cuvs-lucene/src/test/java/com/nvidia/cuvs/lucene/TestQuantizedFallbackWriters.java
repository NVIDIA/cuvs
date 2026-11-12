/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.graphOf;
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
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.AfterClass;
import org.junit.BeforeClass;

/**
 * Checks that the accelerated HNSW codecs build the graph on the CPU when cuVS is not available,
 * and that they read back what their CPU fallback wrote.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestQuantizedFallbackWriters extends LuceneTestCase {

  private static final int NUM_DOCS = 300;
  // Enough vectors for Lucene to build a graph, split into segments. Since Lucene 10.4, which
  // builds one only over more than 100 * ln(n) vectors, the segments are too small for one, so
  // only the merge builds it; earlier releases also build one when they flush each segment.
  private static final int MERGED_DOCS = 1000;
  private static final int DOCS_PER_SEGMENT = 250;
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

  /**
   * Merges segments with the CPU fallback into one large enough for a graph. Since Lucene 10.5, the
   * fallback builds the graph of a merged segment from the flat vectors it has just written, which
   * it reads back with the flat format the codec passes it.
   */
  public void testFallbackWritersMerge() throws Exception {
    AcceleratedHNSWParams params =
        new AcceleratedHNSWParams.Builder().withMaxConn(16).withBeamWidth(100).build();
    for (Codec codec :
        List.of(
            CuVSCodecs.acceleratedHNSW(params),
            CuVSCodecs.acceleratedHNSWScalarQuantized(params),
            CuVSCodecs.acceleratedHNSWBinaryQuantized(params))) {
      float[][] vectors = randomVectors(new Random(random().nextLong()), MERGED_DOCS);
      try (Directory dir = newDirectory()) {
        int fallbacks = writeIndex(dir, codec, vectors, DOCS_PER_SEGMENT, true);
        assertTrue(codec.getName() + " did not fall back to the CPU", fallbacks > 0);
        try (DirectoryReader reader = DirectoryReader.open(dir)) {
          assertEquals(codec.getName(), 1, reader.leaves().size());
          assertEquals(
              codec.getName() + ": the merged segment should have a graph over every vector",
              MERGED_DOCS,
              graphOf(reader.leaves().get(0).reader(), "vector").size());
        }
        assertVectorsFindThemselves(dir, codec, vectors);
      }
    }
  }

  private void checkWritesAndReads(Codec codec) throws Exception {
    float[][] vectors = randomVectors(new Random(random().nextLong()), NUM_DOCS);
    try (Directory dir = newDirectory()) {
      int fallbacks = writeIndex(dir, codec, vectors, IndexWriterConfig.DISABLE_AUTO_FLUSH, false);
      assertTrue(codec.getName() + " did not fall back to the CPU", fallbacks > 0);
      assertVectorsFindThemselves(dir, codec, vectors);
    }
  }

  private static float[][] randomVectors(Random random, int count) {
    float[][] vectors = new float[count][DIMENSIONS];
    for (float[] vector : vectors) {
      for (int d = 0; d < DIMENSIONS; d++) {
        vector[d] = random.nextFloat();
      }
    }
    return vectors;
  }

  /**
   * Indexes one document per vector, with its position as the stored "id", flushing every {@code
   * maxBufferedDocs} documents and then merging into one segment if {@code forceMerge}, which must
   * fall back to the CPU too, and returns how many times the codec logged that it fell back.
   */
  private static int writeIndex(
      Directory dir, Codec codec, float[][] vectors, int maxBufferedDocs, boolean forceMerge)
      throws Exception {
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
    // Flushes and merges on this thread, where cuVS is unavailable.
    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(codec)
            .setUseCompoundFile(false)
            .setMaxBufferedDocs(maxBufferedDocs)
            .setMergePolicy(forceMerge ? new TieredMergePolicy() : NoMergePolicy.INSTANCE)
            .setMergeScheduler(new SerialMergeScheduler());
    try (IndexWriter writer = new IndexWriter(dir, config)) {
      for (int i = 0; i < vectors.length; i++) {
        Document doc = new Document();
        doc.add(new StoredField("id", i));
        doc.add(new KnnFloatVectorField("vector", vectors[i], EUCLIDEAN));
        writer.addDocument(doc);
      }
      if (forceMerge) {
        // Flush the last segment first, so that a fallback logged from here on comes from the
        // merge.
        writer.flush();
        int beforeMerge = fallbacks.get();
        writer.forceMerge(1);
        assertTrue(codec.getName() + " did not merge on the CPU", fallbacks.get() > beforeMerge);
      }
    } finally {
      logger.removeHandler(countFallbacks);
    }
    return fallbacks.get();
  }

  /** Checks that at least 90% of the vectors are among their own top-k search results. */
  private static void assertVectorsFindThemselves(Directory dir, Codec codec, float[][] vectors)
      throws Exception {
    try (DirectoryReader reader = DirectoryReader.open(dir)) {
      assertEquals(codec.getName(), vectors.length, reader.numDocs());
      IndexSearcher searcher = new IndexSearcher(reader);
      int found = 0;
      for (int i = 0; i < vectors.length; i++) {
        ScoreDoc[] hits =
            searcher.search(new KnnFloatVectorQuery("vector", vectors[i], TOP_K), TOP_K).scoreDocs;
        for (ScoreDoc hit : hits) {
          if (searcher.storedFields().document(hit.doc).getField("id").numericValue().intValue()
              == i) {
            found++;
            break;
          }
        }
      }
      assertTrue(
          codec.getName()
              + ": only "
              + found
              + " of "
              + vectors.length
              + " vectors found themselves",
          found >= vectors.length * 0.9);
    }
  }
}

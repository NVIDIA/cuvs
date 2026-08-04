/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.generateDataset;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import com.nvidia.cuvs.spi.CuVSProvider;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * End-to-end concurrency test for the filter-bitset path of multi-segment GPU search. Many threads
 * issue filtered {@link GPUKnnFloatVectorQuery} searches across a rotating set of distinct filters
 * (more than the test's deliberately small cache budget spans) so the shared filter cache builds,
 * caches, evicts, and reuses handles under contention. Every returned hit must satisfy its filter,
 * which validates that reference-counted eviction never frees a handle still in use.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestMultiSegmentGPUFilterConcurrency extends LuceneTestCase {

  // Use the real GPU-search codec (returns the CuVS format directly, not wrapped) so the reader is
  // a CuVS2510GPUVectorsReader and GPUKnnFloatVectorQuery takes the multi-partition GPU path rather
  // than falling back to per-segment search.
  private static Codec codec;
  private static final String VECTOR_FIELD = "vectors";
  private static final String CATEGORY_FIELD = "cat";
  private static final long TEST_CACHE_MAX_BYTES = 4 * 1024;
  private static final FilterBitsetCacheConfig TEST_CACHE_CONFIG =
      new FilterBitsetCacheConfig(true, TEST_CACHE_MAX_BYTES);
  private static final int NUM_CATEGORIES = 24;

  private static Directory directory;
  private static DirectoryReader reader;
  private static IndexSearcher searcher;
  private static float[][] queryVectors;
  private static int topK;
  private static List<Query> filters;
  private static List<Set<Integer>> acceptedDocsPerFilter;

  @BeforeClass
  public static void beforeClass() throws Exception {
    try {
      CuVSProvider.provider().enableRMMAsyncMemory();
    } catch (UnsupportedOperationException unsupported) {
      assumeTrue("cuVS not supported: " + unsupported.getMessage(), false);
    }
    assumeTrue("cuVS not supported", isSupported());
    codec =
        new CuVS2510GPUSearchCodec(
            new GPUSearchParams.Builder().build(), TEST_CACHE_CONFIG);

    int datasetSize = 2000;
    int dimensions = 128;
    int numQueries = 64;
    topK = 10;

    directory = newDirectory(new ByteBuffersDirectory());
    // Disable merges so several commits yield several GPU segments -> multi-partition search.
    IndexWriterConfig config =
        new IndexWriterConfig().setCodec(codec).setMergePolicy(NoMergePolicy.INSTANCE);
    float[][] dataset = generateDataset(random(), datasetSize, dimensions);

    try (IndexWriter writer = new IndexWriter(directory, config)) {
      final int commitEvery = datasetSize / 4;
      for (int i = 0; i < datasetSize; i++) {
        Document doc = new Document();
        doc.add(new StringField("id", String.valueOf(i), Field.Store.YES));
        doc.add(new StringField(CATEGORY_FIELD, "c" + (i % NUM_CATEGORIES), Field.Store.NO));
        doc.add(new KnnFloatVectorField(VECTOR_FIELD, dataset[i], EUCLIDEAN));
        writer.addDocument(doc);
        if ((i + 1) % commitEvery == 0) {
          writer.commit();
        }
      }
      writer.commit();
    }

    // DirectoryReader resolves the stored codec name through SPI. Give that singleton the same
    // deliberately small cache budget while it constructs this test's segment readers, then restore
    // its previous format so the test does not alter later reader construction in this JVM.
    CuVS2510GPUSearchCodec spiCodec =
        (CuVS2510GPUSearchCodec) Codec.forName("CuVS2510GPUSearchCodec");
    KnnVectorsFormat previousFormat = spiCodec.knnVectorsFormat();
    spiCodec.setKnnFormat(
        new CuVS2510GPUVectorsFormat(new GPUSearchParams.Builder().build(), TEST_CACHE_CONFIG));
    try {
      reader = DirectoryReader.open(directory);
    } finally {
      spiCodec.setKnnFormat(previousFormat);
    }
    searcher = new IndexSearcher(reader);
    queryVectors = generateDataset(random(), numQueries, dimensions);

    // One distinct filter per category, plus the set of global doc IDs each filter accepts.
    filters = new ArrayList<>(NUM_CATEGORIES);
    acceptedDocsPerFilter = new ArrayList<>(NUM_CATEGORIES);
    for (int c = 0; c < NUM_CATEGORIES; c++) {
      Query filter = new TermQuery(new Term(CATEGORY_FIELD, "c" + c));
      filters.add(filter);
      Set<Integer> accepted = new HashSet<>();
      for (ScoreDoc sd : searcher.search(filter, datasetSize).scoreDocs) {
        accepted.add(sd.doc);
      }
      acceptedDocsPerFilter.add(accepted);
    }
  }

  @Test
  public void testConcurrentFilteredSearchAcrossManyFilters() throws Exception {
    final int numThreads = 6;
    final int iterationsPerThread = 150;

    List<Throwable> errors = new CopyOnWriteArrayList<>();
    AtomicInteger nonEmptyResults = new AtomicInteger();

    ExecutorService pool = Executors.newFixedThreadPool(numThreads);
    CountDownLatch go = new CountDownLatch(1);
    List<Future<?>> tasks = new ArrayList<>();
    try {
      for (int t = 0; t < numThreads; t++) {
        tasks.add(
            pool.submit(
                () -> {
                  ThreadLocalRandom rnd = ThreadLocalRandom.current();
                  go.await();
                  for (int i = 0; i < iterationsPerThread; i++) {
                    int c = rnd.nextInt(NUM_CATEGORIES);
                    Query filter = filters.get(c);
                    Set<Integer> accepted = acceptedDocsPerFilter.get(c);
                    float[] q = queryVectors[rnd.nextInt(queryVectors.length)];

                    GPUKnnFloatVectorQuery query =
                        new GPUKnnFloatVectorQuery(VECTOR_FIELD, q, topK, filter, topK, 1);
                    ScoreDoc[] hits = searcher.search(query, topK).scoreDocs;
                    if (hits.length > 0) {
                      nonEmptyResults.incrementAndGet();
                    }
                    for (ScoreDoc hit : hits) {
                      if (!accepted.contains(hit.doc)) {
                        throw new AssertionError(
                            "hit " + hit.doc + " not accepted by filter c" + c);
                      }
                    }
                  }
                  return null;
                }));
      }
      go.countDown();
      for (Future<?> f : tasks) {
        try {
          f.get(120, TimeUnit.SECONDS);
        } catch (Throwable t) {
          errors.add(t);
        }
      }
    } finally {
      pool.shutdownNow();
      pool.awaitTermination(10, TimeUnit.SECONDS);
    }

    if (!errors.isEmpty()) {
      throw new AssertionError(
          errors.size() + " search thread(s) failed; first: " + errors.get(0), errors.get(0));
    }
    assertTrue("expected at least some non-empty filtered results", nonEmptyResults.get() > 0);
  }

  @AfterClass
  public static void afterClass() throws IOException {
    if (reader != null) reader.close();
    if (directory != null) directory.close();
    reader = null;
    directory = null;
    searcher = null;
    filters = null;
    acceptedDocsPerFilter = null;
    queryVectors = null;
  }
}

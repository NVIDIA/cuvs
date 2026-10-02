/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.util.concurrent.atomic.AtomicInteger;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

/** Tests the per-segment search of {@link CompatKnnFloatVectorQuery}, which needs no GPU. */
public class TestCompatKnnFloatVectorQuery extends LuceneTestCase {

  private static final String FIELD = "f";

  /**
   * Each segment must be searched once. Since Lucene 10.3, an optimistic collector manager makes
   * {@code rewrite} search again every segment whose hits all beat the global top-k, expecting a
   * wider second search; the query ignores the manager, so that search would only repeat the first,
   * which for the GPU query is a second identical CAGRA search.
   */
  @Test
  public void testSearchesEachSegmentOnce() throws Exception {
    int k = 5;
    try (Directory dir = newDirectory()) {
      IndexWriterConfig cfg = new IndexWriterConfig().setMergePolicy(NoMergePolicy.INSTANCE);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        // One segment holds every nearest neighbor of the query, the other only distant vectors,
        // so all of the first segment's hits make the global top-k.
        addDocs(w, 0f, 20);
        w.commit();
        addDocs(w, 1000f, 20);
        w.commit();
      }

      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        assertEquals(2, reader.leaves().size());
        CountingQuery query = new CountingQuery(new float[] {0f, 0f}, k);
        assertEquals(k, new IndexSearcher(reader).search(query, k).scoreDocs.length);
        assertEquals("segment searches", reader.leaves().size(), query.searches.get());
      }
    }
  }

  private static void addDocs(IndexWriter w, float offset, int count) throws Exception {
    for (int i = 0; i < count; i++) {
      Document doc = new Document();
      doc.add(
          new KnnFloatVectorField(
              FIELD, new float[] {offset + i, offset + i}, VectorSimilarityFunction.EUCLIDEAN));
      w.addDocument(doc);
    }
  }

  /** Counts the per-segment searches. */
  private static final class CountingQuery extends CompatKnnFloatVectorQuery {

    final AtomicInteger searches = new AtomicInteger();

    CountingQuery(float[] target, int k) {
      super(FIELD, target, k, null);
    }

    @Override
    KnnCollector newPerLeafCollector(int visitedLimit) {
      searches.incrementAndGet();
      return new TopKnnCollector(k, visitedLimit);
    }
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.QueryTimeout;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

/** Tests the per-segment search of {@link CompatKnnFloatVectorQuery}, which needs no GPU. */
public class TestCompatKnnFloatVectorQuery extends LuceneTestCase {

  private static final String FIELD = "f";
  private static final String TAG_FIELD = "tag";
  // A document's position in its segment, for filters accepting a given number per segment.
  private static final String POSITION_FIELD = "position";

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
        CountingQuery query = new CountingQuery(new float[] {0f, 0f}, k, null);
        assertEquals(k, new IndexSearcher(reader).search(query, k).scoreDocs.length);
        assertEquals("segment searches", reader.leaves().size(), query.searches.get());
      }
    }
  }

  /**
   * A filtered segment search must return as many hits as Lucene expects from it. Since Lucene 10.3
   * that is often more than k, and Lucene throws away a search that returns fewer and searches the
   * segment exactly instead, which for the GPU query wastes the GPU search.
   */
  @Test
  public void testFilteredSearchResultsAreKept() throws Exception {
    int k = 5;
    try (Directory dir = newDirectory()) {
      IndexWriterConfig cfg = new IndexWriterConfig().setMergePolicy(NoMergePolicy.INSTANCE);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        // Two equal segments, each with far more accepted documents than perLeafTopK.
        addDocs(w, 0f, 200);
        w.commit();
        addDocs(w, 1000f, 200);
        w.commit();
      }

      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        assertEquals(2, reader.leaves().size());
        // A filter Lucene cannot rewrite away, unlike MatchAllDocsQuery, that accepts every
        // document.
        Query filter = new TermQuery(new Term(TAG_FIELD, "y"));
        CountingQuery query = new CountingQuery(new float[] {0f, 0f}, k, filter);
        assertTrue(
            "perLeafTopK " + query.perLeafTopK(reader.leaves().get(0)),
            query.perLeafTopK(reader.leaves().get(0)) < 200);
        assertEquals(k, new IndexSearcher(reader).search(query, k).scoreDocs.length);
        assertEquals("segment searches", reader.leaves().size(), query.searches.get());
        assertEquals("exact searches", 0, query.exactSearches.get());
      }
    }
  }

  /**
   * {@link CompatKnnFloatVectorQuery#perLeafTopK} must be the boundary Lucene itself applies: a
   * segment with that many accepted documents is searched exactly, one with one more
   * approximately. Lucene computes it in a private method, so this catches a release that changes
   * it.
   */
  @Test
  public void testPerLeafTopKIsLucenesExactSearchBoundary() throws Exception {
    int k = 5;
    try (Directory dir = newDirectory()) {
      IndexWriterConfig cfg = new IndexWriterConfig().setMergePolicy(NoMergePolicy.INSTANCE);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        addDocs(w, 0f, 200);
        w.commit();
        addDocs(w, 1000f, 200);
        w.commit();
      }

      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        assertEquals(2, reader.leaves().size());
        float[] target = new float[] {0f, 0f};
        // Both segments are the same size, so they have the same perLeafTopK.
        int perLeafTopK = new CountingQuery(target, k, null).perLeafTopK(reader.leaves().get(0));
        IndexSearcher searcher = new IndexSearcher(reader);

        CountingQuery atBoundary =
            new CountingQuery(
                target, k, IntPoint.newRangeQuery(POSITION_FIELD, 0, perLeafTopK - 1));
        searcher.search(atBoundary, k);
        assertEquals("approximate searches at " + perLeafTopK, 0, atBoundary.searches.get());
        assertEquals("exact searches at " + perLeafTopK, 2, atBoundary.exactSearches.get());

        CountingQuery aboveBoundary =
            new CountingQuery(target, k, IntPoint.newRangeQuery(POSITION_FIELD, 0, perLeafTopK));
        searcher.search(aboveBoundary, k);
        assertEquals(
            "approximate searches at " + (perLeafTopK + 1), 2, aboveBoundary.searches.get());
      }
    }
  }

  private static void addDocs(IndexWriter w, float offset, int count) throws Exception {
    for (int i = 0; i < count; i++) {
      Document doc = new Document();
      doc.add(
          new KnnFloatVectorField(
              FIELD, new float[] {offset + i, offset + i}, VectorSimilarityFunction.EUCLIDEAN));
      doc.add(new StringField(TAG_FIELD, "y", Field.Store.NO));
      doc.add(new IntPoint(POSITION_FIELD, i));
      w.addDocument(doc);
    }
  }

  /** Counts the per-segment approximate and exact searches. */
  private static final class CountingQuery extends CompatKnnFloatVectorQuery {

    final AtomicInteger searches = new AtomicInteger();
    final AtomicInteger exactSearches = new AtomicInteger();

    CountingQuery(float[] target, int k, Query filter) {
      super(FIELD, target, k, filter);
    }

    @Override
    KnnCollector newPerLeafCollector(LeafReaderContext context, int k, int visitedLimit) {
      searches.incrementAndGet();
      return new TopKnnCollector(k, visitedLimit);
    }

    @Override
    protected TopDocs exactSearch(
        LeafReaderContext context, DocIdSetIterator acceptIterator, QueryTimeout queryTimeout)
        throws IOException {
      exactSearches.incrementAndGet();
      return super.exactSearch(context, acceptIterator, queryTimeout);
    }
  }
}

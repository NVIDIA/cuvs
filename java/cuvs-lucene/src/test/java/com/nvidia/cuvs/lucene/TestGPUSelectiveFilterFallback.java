/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.generateDataset;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;
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
import org.apache.lucene.search.KnnFloatVectorQuery;
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
 * Asserts that {@link GPUKnnFloatVectorQuery} keeps Lucene's exactness contract for selective
 * filters (issue #2522): a filter leaving no more than {@code k} candidates in a segment must be
 * answered by Lucene's exact search, not by an approximate CAGRA search that can miss those
 * candidates.
 *
 * <p>The index is built with {@link NoMergePolicy} and equal-size commits so every segment carries a
 * CAGRA index of the same graph degree — the precondition for {@code rewrite()} to take the
 * multi-partition GPU path at all. Each test therefore also asserts <em>which</em> path ran, by
 * checking for the GPU path's marker query {@code GPUDocAndScoreQuery}; without that, a change that
 * silently sends everything to the CPU would still pass.
 *
 * <p>Selectivity is controlled per segment: field {@code selN} is set on the first {@code N}
 * documents of every segment, so a {@link TermQuery} on it accepts exactly {@code N} documents in
 * each segment.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestGPUSelectiveFilterFallback extends LuceneTestCase {

  private static final String VECTOR_FIELD = "vectors";
  private static final String GPU_PATH_MARKER = "GPUDocAndScoreQuery";
  private static final int DIMENSIONS = 64;
  private static final int SEGMENTS = 4;
  private static final int DOCS_PER_SEGMENT = 500;
  private static final int DATASET_SIZE = SEGMENTS * DOCS_PER_SEGMENT;
  private static final int TOP_K = 10;

  /** Per-segment filter cardinalities indexed as {@code sel<N>}, straddling {@link #TOP_K}. */
  private static final int[] SELECTIVITIES = {1, TOP_K, TOP_K + 1, 200};

  /** Field set only on the documents of the first two segments, leaving the other two empty. */
  private static final String FIRST_HALF_FIELD = "firstHalf";

  private static Directory directory;
  private static DirectoryReader reader;
  private static IndexSearcher searcher;
  private static float[][] dataset;

  @BeforeClass
  public static void beforeClass() throws Exception {
    assumeTrue("cuVS not supported", isSupported());

    directory = newDirectory(new ByteBuffersDirectory());
    dataset = generateDataset(random(), DATASET_SIZE, DIMENSIONS);

    // Disable merges so the commits below yield one GPU segment each -> multi-partition search.
    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(new CuVS2510GPUSearchCodec())
            .setMergePolicy(NoMergePolicy.INSTANCE);
    try (IndexWriter writer = new IndexWriter(directory, config)) {
      for (int i = 0; i < DATASET_SIZE; i++) {
        writer.addDocument(doc(i, i % DOCS_PER_SEGMENT, i < 2 * DOCS_PER_SEGMENT));
        if ((i + 1) % DOCS_PER_SEGMENT == 0) {
          writer.commit();
        }
      }
      writer.commit();
    }

    reader = DirectoryReader.open(directory);
    assertEquals("expected one segment per commit", SEGMENTS, reader.leaves().size());
    searcher = new IndexSearcher(reader);
  }

  @AfterClass
  public static void afterClass() throws Exception {
    if (reader != null) reader.close();
    if (directory != null) directory.close();
    searcher = null;
    reader = null;
    directory = null;
    dataset = null;
  }

  /** A document whose {@code selN} fields mark it as one of the first {@code N} of its segment. */
  private static Document doc(int id, int positionInSegment, boolean firstHalf) {
    Document doc = new Document();
    doc.add(new StringField("id", String.valueOf(id), Field.Store.YES));
    for (int n : SELECTIVITIES) {
      if (positionInSegment < n) {
        doc.add(new StringField("sel" + n, "y", Field.Store.NO));
      }
    }
    if (firstHalf) {
      doc.add(new StringField(FIRST_HALF_FIELD, "y", Field.Store.NO));
    }
    doc.add(new KnnFloatVectorField(VECTOR_FIELD, dataset[id], EUCLIDEAN));
    return doc;
  }

  /**
   * A filter matching a single document per segment is far more selective than {@code k}, so its
   * documents must all come back — the case reported in issue #2522, where the approximate GPU
   * search returned nothing at all.
   */
  @Test
  public void singleMatchPerSegmentReturnsEveryMatch() throws Exception {
    assumeTrue("cuVS not supported", isSupported());
    assertExactFallback(1);
  }

  /** The boundary of Lucene's rule: exactly {@code k} candidates per segment still means exact. */
  @Test
  public void exactlyTopKMatchesPerSegmentIsAnsweredExactly() throws Exception {
    assumeTrue("cuVS not supported", isSupported());
    assertExactFallback(TOP_K);
  }

  /**
   * Just past the boundary the GPU path is allowed again, and just below it the query is exact — the
   * two regimes must disagree about the path while both returning filter-accepted documents.
   */
  @Test
  public void aboveTopKMatchesStaysOnTheGpuPath() throws Exception {
    assumeTrue("cuVS not supported", isSupported());

    float[] queryVector = generateDataset(random(), 1, DIMENSIONS)[0];
    Query filter = new TermQuery(new Term("sel200", "y"));
    // 200 candidates per segment with a generous itopk: CAGRA has ample room to return k hits, so
    // this stays on the GPU path rather than tripping the "fewer than k hits" exact fallback.
    GPUKnnFloatVectorQuery query =
        new GPUKnnFloatVectorQuery(VECTOR_FIELD, queryVector, TOP_K, filter, 64, 1);

    assertTrue(
        "a filter leaving far more than k candidates per segment must use the GPU path",
        searcher.rewrite(query).toString().contains(GPU_PATH_MARKER));

    ScoreDoc[] hits = searcher.search(query, TOP_K).scoreDocs;
    assertEquals(TOP_K, hits.length);
    assertAllAccepted(filter, hits);
  }

  /**
   * A segment whose filter accepts nothing contributes nothing on either path, so it must not drag
   * the whole query onto the exact path: here two segments hold 500 candidates each and two hold
   * none.
   */
  @Test
  public void emptySegmentsDoNotForceExactSearch() throws Exception {
    assumeTrue("cuVS not supported", isSupported());

    float[] queryVector = generateDataset(random(), 1, DIMENSIONS)[0];
    Query filter = new TermQuery(new Term(FIRST_HALF_FIELD, "y"));
    GPUKnnFloatVectorQuery query =
        new GPUKnnFloatVectorQuery(VECTOR_FIELD, queryVector, TOP_K, filter, 64, 1);

    assertTrue(
        "segments with no accepted documents must not force the exact path",
        searcher.rewrite(query).toString().contains(GPU_PATH_MARKER));

    ScoreDoc[] hits = searcher.search(query, TOP_K).scoreDocs;
    assertEquals(TOP_K, hits.length);
    assertAllAccepted(filter, hits);
  }

  /** The rule is about filter cardinality, not segment count: it holds on a single segment too. */
  @Test
  public void singleSegmentSelectiveFilterIsAnsweredExactly() throws Exception {
    assumeTrue("cuVS not supported", isSupported());

    float[][] vectors = generateDataset(random(), DOCS_PER_SEGMENT, DIMENSIONS);
    try (Directory dir = newDirectory(new ByteBuffersDirectory())) {
      IndexWriterConfig config =
          new IndexWriterConfig()
              .setCodec(new CuVS2510GPUSearchCodec())
              .setMergePolicy(NoMergePolicy.INSTANCE);
      try (IndexWriter writer = new IndexWriter(dir, config)) {
        for (int i = 0; i < DOCS_PER_SEGMENT; i++) {
          Document doc = new Document();
          doc.add(new StringField("id", String.valueOf(i), Field.Store.YES));
          doc.add(new KnnFloatVectorField(VECTOR_FIELD, vectors[i], EUCLIDEAN));
          writer.addDocument(doc);
        }
        writer.commit();
      }

      try (DirectoryReader singleSegment = DirectoryReader.open(dir)) {
        assertEquals("expected a single segment", 1, singleSegment.leaves().size());
        IndexSearcher singleSearcher = new IndexSearcher(singleSegment);

        // Query with one vector while filtering to a single unrelated document: an approximate
        // search can miss it, an exact one cannot. Sweep the document the filter selects, since
        // whether the approximate search happens to reach it depends on the graph.
        for (int matchingId = 0; matchingId < DOCS_PER_SEGMENT; matchingId += 10) {
          Query filter = new TermQuery(new Term("id", String.valueOf(matchingId)));
          GPUKnnFloatVectorQuery query =
              new GPUKnnFloatVectorQuery(VECTOR_FIELD, vectors[0], TOP_K, filter, TOP_K, 1);

          assertFalse(
              "a single-document filter must not use the GPU path",
              singleSearcher.rewrite(query).toString().contains(GPU_PATH_MARKER));

          ScoreDoc[] hits = singleSearcher.search(query, TOP_K).scoreDocs;
          assertEquals(
              "document " + matchingId + " matches the filter and must be returned",
              1,
              hits.length);
          assertEquals(
              String.valueOf(matchingId),
              singleSegment.storedFields().document(hits[0].doc).get("id"));
        }
      }
    }
  }

  /**
   * Runs a filter of per-segment cardinality {@code n} ({@code n <= TOP_K}) and asserts it left the
   * GPU path and produced exactly what Lucene's own {@link KnnFloatVectorQuery} produces — same
   * documents, same scores, since both are then served by the same exact search.
   */
  private void assertExactFallback(int n) throws IOException {
    float[] queryVector = generateDataset(random(), 1, DIMENSIONS)[0];
    Query filter = new TermQuery(new Term("sel" + n, "y"));

    GPUKnnFloatVectorQuery gpuQuery =
        new GPUKnnFloatVectorQuery(VECTOR_FIELD, queryVector, TOP_K, filter, TOP_K, 1);
    assertFalse(
        "a filter leaving no more than k candidates per segment must not use the GPU path",
        searcher.rewrite(gpuQuery).toString().contains(GPU_PATH_MARKER));

    ScoreDoc[] hits = searcher.search(gpuQuery, TOP_K).scoreDocs;
    assertEquals(
        "every accepted document up to k must be returned",
        Math.min(TOP_K, n * SEGMENTS),
        hits.length);
    assertAllAccepted(filter, hits);

    ScoreDoc[] expected =
        searcher.search(new KnnFloatVectorQuery(VECTOR_FIELD, queryVector, TOP_K, filter), TOP_K)
            .scoreDocs;
    assertEquals("same number of hits as Lucene", expected.length, hits.length);
    for (int i = 0; i < expected.length; i++) {
      assertEquals("hit " + i + " doc", expected[i].doc, hits[i].doc);
      assertEquals("hit " + i + " score", expected[i].score, hits[i].score, 0.0f);
    }
  }

  /** Every returned document must pass the filter. */
  private void assertAllAccepted(Query filter, ScoreDoc[] hits) throws IOException {
    Set<Integer> accepted = new HashSet<>();
    for (ScoreDoc sd : searcher.search(filter, DATASET_SIZE).scoreDocs) {
      accepted.add(sd.doc);
    }
    for (ScoreDoc hit : hits) {
      assertTrue(
          "returned doc " + hit.doc + " is not accepted by the filter", accepted.contains(hit.doc));
    }
  }
}

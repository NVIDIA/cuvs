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
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.util.Version;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Asserts that {@link GPUKnnFloatVectorQuery} keeps Lucene's exactness contract for selective
 * filters (issue #2522): a filter leaving no more than {@code k} candidates in a segment, or {@link
 * CompatKnnFloatVectorQuery#perLeafTopK} if that is larger, must be answered by Lucene's per-segment
 * search, not by the multi-partition CAGRA search. Lucene searches a segment with no more than
 * {@code perLeafTopK} candidates exactly, which an approximate search cannot match, and checks
 * that the approximate search of a segment with more returned enough hits, which the
 * multi-partition search cannot check per segment.
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

  /** A document's position in its segment, for filters of any per-segment cardinality. */
  private static final String POSITION_FIELD = "position";

  /** The number of a document's segment, for filters accepting documents of one segment only. */
  private static final String SEGMENT_FIELD = "segment";

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
            .setCodec(CuVSCodecs.gpuSearch())
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
    doc.add(new IntPoint(POSITION_FIELD, positionInSegment));
    doc.add(new IntPoint(SEGMENT_FIELD, id / DOCS_PER_SEGMENT));
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

  /**
   * Exactly {@code k} candidates per segment means exact on every release: the boundary on Lucene
   * 10.2, and below it on 10.3, where {@code perLeafTopK} is larger here.
   */
  @Test
  public void exactlyTopKMatchesPerSegmentIsAnsweredExactly() throws Exception {
    assumeTrue("cuVS not supported", isSupported());
    assertExactFallback(TOP_K);
  }

  /**
   * The boundary of the fallback: exactly {@code max(k, perLeafTopK)} candidates per segment still
   * means exact.
   */
  @Test
  public void fallbackBoundaryMatchesPerSegmentIsAnsweredExactly() throws Exception {
    assumeTrue("cuVS not supported", isSupported());
    int boundary = fallbackBoundary(TOP_K);
    assertExactFallback(IntPoint.newRangeQuery(POSITION_FIELD, 0, boundary - 1), boundary);
  }

  /**
   * One candidate past the boundary in a segment allows the GPU path. The other segments hold 200
   * candidates each, so that CAGRA surely finds k hits and the result-count check cannot fall back
   * either.
   */
  @Test
  public void pastFallbackBoundaryUsesTheGpuPath() throws Exception {
    assumeTrue("cuVS not supported", isSupported());
    int n = fallbackBoundary(TOP_K) + 1;
    Query firstSegment =
        new BooleanQuery.Builder()
            .add(IntPoint.newExactQuery(SEGMENT_FIELD, 0), BooleanClause.Occur.FILTER)
            .add(IntPoint.newRangeQuery(POSITION_FIELD, 0, n - 1), BooleanClause.Occur.FILTER)
            .build();
    Query otherSegments =
        new BooleanQuery.Builder()
            .add(IntPoint.newRangeQuery(SEGMENT_FIELD, 1, SEGMENTS - 1), BooleanClause.Occur.FILTER)
            .add(new TermQuery(new Term("sel200", "y")), BooleanClause.Occur.FILTER)
            .build();
    Query filter =
        new BooleanQuery.Builder()
            .add(firstSegment, BooleanClause.Occur.SHOULD)
            .add(otherSegments, BooleanClause.Occur.SHOULD)
            .build();
    float[] queryVector = generateDataset(random(), 1, DIMENSIONS)[0];
    GPUKnnFloatVectorQuery query =
        new GPUKnnFloatVectorQuery(VECTOR_FIELD, queryVector, TOP_K, filter, 64, 1);

    assertTrue(
        n + " candidates in a segment, one past the boundary, must allow the GPU path",
        searcher.rewrite(query).toString().contains(GPU_PATH_MARKER));
    ScoreDoc[] hits = searcher.search(query, TOP_K).scoreDocs;
    assertEquals(TOP_K, hits.length);
    assertAllAccepted(filter, hits);
  }

  /**
   * Since Lucene 10.3, with a large {@code k} over several segments, {@code perLeafTopK} is less
   * than {@code k}. Lucene then searches a segment with between the two candidates approximately,
   * but searches it again exactly if that returns fewer than {@code perLeafTopK} hits, which the
   * multi-partition search cannot check per segment. Such a segment must still send the query to
   * Lucene's per-segment search, which returns at least {@code perLeafTopK} hits.
   */
  @Test
  public void candidatesBetweenPerLeafTopKAndTopKUseThePerSegmentPath() throws Exception {
    assumeTrue("cuVS not supported", isSupported());
    int k = 100;
    assumeTrue(
        "perLeafTopK is less than k only since Lucene 10.3",
        Version.LATEST.onOrAfter(Version.fromBits(10, 3, 0)));
    int perLeafTopK = perLeafTopK(k);
    assertTrue("perLeafTopK " + perLeafTopK + " for k " + k + " over 4 segments", perLeafTopK < k);
    int n = perLeafTopK + 1;
    float[] queryVector = generateDataset(random(), 1, DIMENSIONS)[0];
    Query filter =
        new BooleanQuery.Builder()
            .add(IntPoint.newExactQuery(SEGMENT_FIELD, 0), BooleanClause.Occur.FILTER)
            .add(IntPoint.newRangeQuery(POSITION_FIELD, 0, n - 1), BooleanClause.Occur.FILTER)
            .build();
    GPUKnnFloatVectorQuery query =
        new GPUKnnFloatVectorQuery(VECTOR_FIELD, queryVector, k, filter, 128, 1);

    assertFalse(
        n + " candidates, perLeafTopK " + perLeafTopK + ", k " + k + " must not use the GPU path",
        searcher.rewrite(query).toString().contains(GPU_PATH_MARKER));
    ScoreDoc[] hits = searcher.search(query, k).scoreDocs;
    assertTrue(
        hits.length + " hits, expected " + perLeafTopK + " to " + n,
        hits.length >= perLeafTopK && hits.length <= n);
    assertAllAccepted(filter, hits);
  }

  /** Returns {@code max(k, perLeafTopK)} for the equal-size segments of this index. */
  private static int fallbackBoundary(int k) {
    return Math.max(k, perLeafTopK(k));
  }

  /**
   * Returns {@code perLeafTopK} for the equal-size segments of this index. It does not depend on
   * the query vector, so this uses a zero vector rather than drawing a random one.
   */
  private static int perLeafTopK(int k) {
    return new GPUKnnFloatVectorQuery(VECTOR_FIELD, new float[DIMENSIONS], k, null, k, 1)
        .perLeafTopK(reader.leaves().get(0));
  }

  /**
   * Far past the boundary the query uses the GPU path and returns k filter-accepted documents.
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
              .setCodec(CuVSCodecs.gpuSearch())
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
   * Runs a filter of per-segment cardinality {@code n}, no more than {@code perLeafTopK} so that
   * Lucene searches every segment exactly, and asserts it left the GPU path and produced exactly
   * what Lucene's own {@link KnnFloatVectorQuery} produces — same documents, same scores, since both
   * are then served by the same exact search.
   */
  private void assertExactFallback(int n) throws IOException {
    assertExactFallback(new TermQuery(new Term("sel" + n, "y")), n);
  }

  /** Like {@link #assertExactFallback(int)}, with a filter accepting {@code n} per segment. */
  private void assertExactFallback(Query filter, int n) throws IOException {
    assertTrue(
        n + " candidates per segment must be searched exactly, so no more than perLeafTopK",
        n <= perLeafTopK(TOP_K));
    float[] queryVector = generateDataset(random(), 1, DIMENSIONS)[0];

    GPUKnnFloatVectorQuery gpuQuery =
        new GPUKnnFloatVectorQuery(VECTOR_FIELD, queryVector, TOP_K, filter, TOP_K, 1);
    assertFalse(
        "a filter leaving no more than max(k, perLeafTopK) candidates per segment must not use the"
            + " GPU path",
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

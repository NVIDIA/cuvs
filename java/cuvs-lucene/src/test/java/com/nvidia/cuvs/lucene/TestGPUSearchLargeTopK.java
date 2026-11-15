/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.generateDataset;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import com.nvidia.cuvs.CagraSearchParams.SearchAlgo;
import com.nvidia.cuvs.lucene.CuVS2510GPUVectorsWriter.IndexType;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.util.Bits;
import org.junit.Test;

/**
 * Searches a GPU index for more hits than CAGRA returns from one segment, {@link
 * CuVS2510GPUVectorsReader#MAX_CAGRA_K}. Since Lucene 10.3, Lucene asks a segment for as many hits
 * as it expects from it, which for a segment holding most of the index exceeds {@code k}, and so
 * can exceed that limit even when {@code k} does not.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestGPUSearchLargeTopK extends LuceneTestCase {

  private static final String VECTOR_FIELD = "vectors";
  private static final int DIMENSIONS = 8;

  /** The visit limit Lucene passes to a segment search when it applies no filter. */
  private static final int UNFILTERED = Integer.MAX_VALUE;

  @Test
  public void largeSegmentWithLargeTopKIsSearched() throws Exception {
    assumeTrue("cuVS not supported", isSupported());
    // A segment holding 90% of the documents: with k = 1000, Lucene 10.3 asks it for about 1052.
    int k = 1000;
    try (Directory dir = newDirectory(new ByteBuffersDirectory());
        DirectoryReader reader = index(dir, CuVSCodecs.gpuSearch(), 9000, 1000)) {
      float[] query = generateDataset(random(), 1, DIMENSIONS)[0];
      int hits =
          new IndexSearcher(reader)
              .search(new KnnFloatVectorQuery(VECTOR_FIELD, query, k), k)
              .scoreDocs
              .length;
      assertEquals(k, hits);
    }
  }

  /**
   * Searching all segments at once has no per-segment limit: two segments return more hits than
   * twice {@link CuVS2510GPUVectorsReader#MAX_CAGRA_K}, which searching them one at a time could not.
   */
  @Test
  public void gpuQueryReturnsMoreHitsThanCagraLimit() throws Exception {
    assumeTrue("cuVS not supported", isSupported());
    int k = 2500;
    try (Directory dir = newDirectory(new ByteBuffersDirectory());
        DirectoryReader reader = index(dir, CuVSCodecs.gpuSearch(), 6000, 4000)) {
      float[] query = generateDataset(random(), 1, DIMENSIONS)[0];
      int hits =
          new IndexSearcher(reader)
              .search(new GPUKnnFloatVectorQuery(VECTOR_FIELD, query, k, null, 64, 1), k)
              .scoreDocs
              .length;
      assertEquals(k, hits);
    }
  }

  /**
   * A segment with only a CAGRA index returns at most {@link CuVS2510GPUVectorsReader#MAX_CAGRA_K}
   * hits, or {@link GPUKnnFloatVectorQuery#MAX_SINGLE_CTA_ITOPK} with SINGLE_CTA. Lucene discards a
   * filtered search's hits when there are fewer than it expects from the segment, so the query
   * skips that search, and keeps the search where that many are enough.
   */
  @Test
  public void cagraOnlySegmentSkipsFilteredSearchItCannotAnswer() throws Exception {
    assumeTrue("cuVS not supported", isSupported());
    try (Directory dir = newDirectory(new ByteBuffersDirectory());
        DirectoryReader reader = index(dir, CuVSCodecs.gpuSearch(), 9000, 3000)) {
      // On Lucene 10.3, the 3000-document segment is expected to return about 643 hits for
      // k = 1500, and the 9000-document one about 1393, fewer than k but still above the limit;
      // for k = 500, about 280 and 530, the latter above SINGLE_CTA's limit. Lucene 10.2 asks every
      // segment for k, so only k = 1500 goes above the limit there.
      boolean asksForK = LuceneCompat.LUCENE_MAJOR == 10 && LuceneCompat.LUCENE_MINOR < 3;
      assertSkippedBeyondLimit(
          reader, gpuQuery(1500, SearchAlgo.AUTO), CuVS2510GPUVectorsReader.MAX_CAGRA_K, true);
      assertSkippedBeyondLimit(
          reader,
          gpuQuery(500, SearchAlgo.SINGLE_CTA),
          GPUKnnFloatVectorQuery.MAX_SINGLE_CTA_ITOPK,
          !asksForK);
    }
  }

  /**
   * Asserts that each segment is searched or skipped as its expected hit count requires, and, if
   * {@code expectSkip}, that at least one segment is skipped, so that the skip is tested at all.
   */
  private static void assertSkippedBeyondLimit(
      DirectoryReader reader, GPUKnnFloatVectorQuery query, int limit, boolean expectSkip)
      throws Exception {
    int skipped = 0;
    for (LeafReaderContext leaf : reader.leaves()) {
      String message = query + " in a segment of " + leaf.reader().maxDoc();
      int perLeafTopK = query.perLeafTopK(leaf);
      int filteredHits = filteredHits(query, leaf);
      if (perLeafTopK <= limit) {
        assertSearched(
            message,
            perLeafTopK,
            Math.min(Math.max(query.getK(), perLeafTopK), limit),
            filteredHits);
      } else {
        assertEquals(message + " should be skipped", 0, filteredHits);
        skipped++;
      }
      int unfilteredMax = Math.min(query.getK(), limit);
      assertSearched(
          message + " without a filter",
          Math.min(perLeafTopK, unfilteredMax),
          unfilteredMax,
          approximateHits(query, leaf, null, UNFILTERED));
    }
    if (expectSkip) {
      assertTrue(query + " should skip at least one segment", skipped > 0);
    }
  }

  /**
   * A segment with a brute force index returns as many hits as Lucene expects, including more than
   * CAGRA's limit for the search algorithm.
   */
  @Test
  public void bruteForceSegmentAnswersFilteredSearchBeyondCagraLimit() throws Exception {
    assumeTrue("cuVS not supported", isSupported());
    Codec codec =
        CuVSCodecs.gpuSearch(
            new GPUSearchParams.Builder().withIndexType(IndexType.CAGRA_AND_BRUTE_FORCE).build());
    try (Directory dir = newDirectory(new ByteBuffersDirectory());
        DirectoryReader reader = index(dir, codec, 27000, 3000)) {
      // On Lucene 10.3, the 27000-document segment is expected to return about 1051 hits for
      // k = 1000, above the CAGRA limit although k is not, and about 557 for k = 500, above
      // SINGLE_CTA's limit. k = 1500 is above the CAGRA limit on every release.
      for (GPUKnnFloatVectorQuery query :
          new GPUKnnFloatVectorQuery[] {
            gpuQuery(1500, SearchAlgo.AUTO),
            gpuQuery(1000, SearchAlgo.AUTO),
            gpuQuery(500, SearchAlgo.SINGLE_CTA)
          }) {
        for (LeafReaderContext leaf : reader.leaves()) {
          String message = query + " in a segment of " + leaf.reader().maxDoc();
          int perLeafTopK = query.perLeafTopK(leaf);
          assertSearched(
              message, perLeafTopK, Math.max(query.getK(), perLeafTopK), filteredHits(query, leaf));
          assertSearched(
              message + " without a filter",
              Math.min(perLeafTopK, query.getK()),
              query.getK(),
              approximateHits(query, leaf, null, UNFILTERED));
        }
      }
    }
  }

  /**
   * Asserts that a segment search ran and returned enough hits for Lucene to keep them, but no more
   * than collected. CAGRA is approximate and may leave a slot empty, so the exact count can vary.
   */
  private static void assertSearched(String message, int minHits, int maxHits, int hits) {
    assertTrue(
        message + ": " + hits + " hits, expected " + minHits + " to " + maxHits,
        hits >= minHits && hits <= maxHits);
  }

  private static GPUKnnFloatVectorQuery gpuQuery(int k, SearchAlgo searchAlgo) {
    float[] target = generateDataset(random(), 1, DIMENSIONS)[0];
    return new GPUKnnFloatVectorQuery(VECTOR_FIELD, target, k, null, 64, 1, 0, 0, searchAlgo);
  }

  /** Searches a segment the way Lucene does with a filter that accepts every document. */
  private static int filteredHits(GPUKnnFloatVectorQuery query, LeafReaderContext leaf)
      throws Exception {
    int maxDoc = leaf.reader().maxDoc();
    // Lucene's visit limit for a filtered search is the filter's cost plus one.
    return approximateHits(query, leaf, new Bits.MatchAllBits(maxDoc), maxDoc + 1);
  }

  private static int approximateHits(
      GPUKnnFloatVectorQuery query, LeafReaderContext leaf, Bits acceptDocs, int visitedLimit)
      throws Exception {
    return TestQueryCompat.approximateSearch(query, leaf, acceptDocs, visitedLimit)
        .scoreDocs
        .length;
  }

  /** Indexes one segment of random vectors per given size. */
  private static DirectoryReader index(Directory dir, Codec codec, int... segmentSizes)
      throws Exception {
    IndexWriterConfig config =
        new IndexWriterConfig().setCodec(codec).setMergePolicy(NoMergePolicy.INSTANCE);
    try (IndexWriter writer = new IndexWriter(dir, config)) {
      for (int size : segmentSizes) {
        for (float[] vector : generateDataset(random(), size, DIMENSIONS)) {
          Document doc = new Document();
          doc.add(new KnnFloatVectorField(VECTOR_FIELD, vector, EUCLIDEAN));
          writer.addDocument(doc);
        }
        writer.commit();
      }
    }
    DirectoryReader reader = DirectoryReader.open(dir);
    int segments = reader.leaves().size();
    if (segments != segmentSizes.length) {
      reader.close();
      fail("expected " + segmentSizes.length + " segments, got " + segments);
    }
    return reader;
  }
}

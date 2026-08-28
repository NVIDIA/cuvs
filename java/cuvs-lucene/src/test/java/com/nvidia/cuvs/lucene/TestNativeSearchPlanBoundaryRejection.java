/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.generateDataset;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import com.nvidia.cuvs.CagraSearchParams;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.tests.util.TestUtil;
import org.junit.Test;

/**
 * Confirms that a {@code searchWidth}/{@code iTopK} combination within {@link
 * GPUKnnFloatVectorQuery}'s Java-level range but unsupported by native CAGRA is rejected by native
 * CAGRA itself with a clear exception, rather than by this API (which does not attempt to
 * replicate native CAGRA's algorithm- and dataset-dependent hash-table sizing -- see the javadoc on
 * {@link GPUKnnFloatVectorQuery#MAX_ITOPK} and {@link GPUKnnFloatVectorQuery#MAX_SEARCH_WIDTH}).
 *
 * <p>{@code MULTI_CTA} -- which a normal one-query {@code AUTO} search resolves to -- sizes an
 * internal traversal hash table from {@code max(searchWidth, ceil(iTopK / 32)) * max(32,
 * maxIterations)}, and native CAGRA hard-limits that table to a 25-bit index (raft::exception via
 * {@code RAFT_EXPECTS(hash_bitlen <= 25, ...)} in {@code search_plan.cuh}). At the default hashmap
 * fill rate of 0.5, that caps the product at 2^25 * 0.5 = 16,777,216. Setting {@code searchWidth}
 * to {@link GPUKnnFloatVectorQuery#MAX_SEARCH_WIDTH} (4,194,303) alone exceeds that cap by 8x
 * even at the smallest possible multiplier (32), regardless of {@code iTopK}, graph degree, or
 * dataset size -- so this test does not need to reproduce native CAGRA's {@code max_iterations}
 * auto-derivation to reliably trigger the rejection.
 *
 * <p>This class does not cover an oversized {@link GPUKnnFloatVectorQuery#MAX_ITOPK} the same
 * way: empirically, {@code iTopK = Integer.MAX_VALUE} does not fail fast like an oversized {@code
 * searchWidth} does -- it hangs inside the native call indefinitely instead of returning an
 * error, which would make a test asserting on it unsafe to run in CI (no bounded timeout reliably
 * recovers a thread stuck in native code). That hang is itself worth separate investigation.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestNativeSearchPlanBoundaryRejection extends LuceneTestCase {

  private static final String VECTOR_FIELD = "vector";

  @Test
  public void testOversizedSearchWidthRejectedByNativeCagra() throws Exception {
    assumeTrue("cuVS not supported", isSupported());

    Codec codec = TestUtil.alwaysKnnVectorsFormat(new CuVS2510GPUVectorsFormat());
    int datasetSize = 200;
    int dimensions = 32;
    float[][] dataset = generateDataset(random(), datasetSize, dimensions);

    try (Directory dir = newDirectory()) {
      IndexWriterConfig cfg = new IndexWriterConfig().setCodec(codec);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (float[] vector : dataset) {
          Document doc = new Document();
          doc.add(new KnnFloatVectorField(VECTOR_FIELD, vector, EUCLIDEAN));
          w.addDocument(doc);
        }
      }

      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        IndexSearcher searcher = new IndexSearcher(reader);

        // A reasonable, genuinely valid combination: small search width, moderate iTopK. Must
        // execute a native search plan successfully.
        int k = 5;
        GPUKnnFloatVectorQuery validQuery =
            new GPUKnnFloatVectorQuery(
                VECTOR_FIELD,
                dataset[0],
                k,
                null,
                64,
                8,
                0,
                0,
                CagraSearchParams.SearchAlgo.MULTI_CTA);
        assertTrue(searcher.search(validQuery, k).scoreDocs.length > 0);

        // Within this class's own Java-level range (searchWidth <= MAX_SEARCH_WIDTH), but far
        // beyond what native CAGRA's traversal hash table can represent for MULTI_CTA -- which a
        // normal one-query AUTO search (used here, not an explicit MULTI_CTA) resolves to. This
        // must be rejected by native CAGRA when the search plan is actually built -- not silently
        // accepted or left to corrupt/misbehave.
        GPUKnnFloatVectorQuery oversizedQuery =
            new GPUKnnFloatVectorQuery(
                VECTOR_FIELD,
                dataset[0],
                k,
                null,
                64,
                GPUKnnFloatVectorQuery.MAX_SEARCH_WIDTH,
                0,
                0,
                CagraSearchParams.SearchAlgo.AUTO);
        RuntimeException e =
            expectThrows(RuntimeException.class, () -> searcher.search(oversizedQuery, k));
        // Assert on the specific native failure (the hash-table bit-length cap) rather than any
        // RuntimeException, so an unrelated native/CUDA failure cannot make this test pass.
        assertTrue(e.getMessage(), e.getMessage().contains("hash_bitlen"));
        assertTrue(e.getMessage(), e.getMessage().contains("25"));
      }
    }
  }
}

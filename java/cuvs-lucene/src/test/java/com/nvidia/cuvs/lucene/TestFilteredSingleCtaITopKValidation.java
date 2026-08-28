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
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.tests.util.TestUtil;
import org.junit.Test;

/**
 * Regression test: the per-segment fallback search path must re-validate the effective SINGLE_CTA
 * iTopK limit against the value actually sent to native CAGRA, not just the value checked at query
 * construction time.
 *
 * <p>{@link GPUKnnFloatVectorQuery} validates {@code max(iTopK, k)} against the SINGLE_CTA limit
 * (512) at construction. But the per-segment fallback path ({@link
 * CuVS2510GPUVectorsReader#search}) -- used when {@link GPUKnnFloatVectorQuery#rewrite} cannot
 * apply its optimized multi-partition search, e.g. because a segment has no CAGRA index for the
 * field -- raises {@code topK} further, up to {@code min(k + 10, filterCardinality)}, whenever a
 * filter is present. A sufficiently permissive filter can push the value actually sent to native
 * CAGRA above 512 even though the value checked at construction time was within range. This must
 * still be rejected before a native search plan is built.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestFilteredSingleCtaITopKValidation extends LuceneTestCase {

  private static final String VECTOR_FIELD = "vector";
  private static final String INCLUDED_FIELD = "included";

  @Test
  public void testFilterDrivenTopKIncreaseIsRevalidatedAgainstSingleCtaLimit() throws Exception {
    assumeTrue("cuVS not supported", isSupported());

    Codec codec = TestUtil.alwaysKnnVectorsFormat(new CuVS2510GPUVectorsFormat());
    int datasetSize = 1000;
    int dimensions = 32;
    float[][] dataset = generateDataset(random(), datasetSize, dimensions);

    try (Directory dir = newDirectory()) {
      IndexWriterConfig cfg = new IndexWriterConfig().setCodec(codec);
      // Keep segments separate: a single-vector segment falls back to a brute-force index
      // (CuVS2510GPUVectorsWriter's MIN_CAGRA_INDEX_SIZE is 2), giving it no CAGRA index for this
      // field. That forces GPUKnnFloatVectorQuery#rewrite to fall back to the standard per-segment
      // Lucene search path -- the only path that raises topK based on filter cardinality -- for
      // every segment, instead of taking the optimized multi-partition path.
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int i = 0; i < datasetSize; i++) {
          Document doc = new Document();
          doc.add(new StringField(INCLUDED_FIELD, "yes", Field.Store.NO));
          doc.add(new KnnFloatVectorField(VECTOR_FIELD, dataset[i], EUCLIDEAN));
          w.addDocument(doc);
        }
        w.commit();

        Document singleVectorDoc = new Document();
        singleVectorDoc.add(new StringField(INCLUDED_FIELD, "no", Field.Store.NO));
        singleVectorDoc.add(new KnnFloatVectorField(VECTOR_FIELD, dataset[0], EUCLIDEAN));
        w.addDocument(singleVectorDoc);
        w.commit();
      }

      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        IndexSearcher searcher = new IndexSearcher(reader);

        // Matches every document in the 1000-vector segment (cardinality 1000) and none in the
        // single-vector segment.
        Query filter = new TermQuery(new Term(INCLUDED_FIELD, "yes"));

        int k = 503;
        int iTopK = 512; // Passes the constructor-time check: max(iTopK, k) == 512 <= 512.
        GPUKnnFloatVectorQuery query =
            new GPUKnnFloatVectorQuery(
                VECTOR_FIELD,
                dataset[0],
                k,
                filter,
                iTopK,
                1,
                0,
                0,
                CagraSearchParams.SearchAlgo.SINGLE_CTA);

        // On the 1000-vector segment, topK becomes min(k + 10, filterCardinality) = min(513, 1000)
        // = 513, so the effective iTopK sent to native CAGRA is max(512, 513) = 513, exceeding the
        // SINGLE_CTA limit of 512. Assert the specific message (not just the exception type) to
        // confirm this post-filter re-validation fired, rather than some unrelated argument check.
        IllegalArgumentException e =
            expectThrows(IllegalArgumentException.class, () -> searcher.search(query, k));
        assertTrue(e.getMessage(), e.getMessage().contains("SINGLE_CTA"));
        assertTrue(e.getMessage(), e.getMessage().contains("512"));
        assertTrue(e.getMessage(), e.getMessage().contains("513"));
      }
    }
  }
}

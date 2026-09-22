/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.generateDataset;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.util.FixedBitSet;
import org.junit.Test;

/**
 * Exercises the per-segment (single-index) GPU search with a selective prefilter. It drives {@link
 * CuVS2510GPUVectorsReader#search} directly via {@code LeafReader.searchNearestVectors} with a
 * synthesized {@code acceptDocs}, so the path is exercised regardless of any higher-level query
 * routing (which now sends the multi-partition case elsewhere).
 *
 * <p>A highly selective {@code acceptDocs} rejects the trailing ordinals of the segment — the regime
 * where the per-segment prefilter previously mis-sized itself (it passed {@code BitSet.length()},
 * the highest set bit + 1, instead of the vector count) and default-accepted the trailing ordinals,
 * returning filtered-out vectors. Every returned hit must be accepted.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestPerSegmentGPUFilterSearch extends LuceneTestCase {

  private static final String VECTOR_FIELD = "vectors";
  private static final int NUM_CATEGORIES = 24;

  @Test
  public void testSelectiveFilterDoesNotLeakTrailingOrdinals() throws Exception {
    assumeTrue("cuVS not supported", isSupported());

    // A segment size that is not a multiple of 32 exercises the last partial uint32 word of the
    // prefilter bitset, which is where the trailing ordinals used to leak.
    final int datasetSize = 500;
    final int dimensions = 128;
    final int topK = 10;

    try (Directory directory = newDirectory(new ByteBuffersDirectory())) {
      float[][] dataset = generateDataset(random(), datasetSize, dimensions);
      IndexWriterConfig config = new IndexWriterConfig().setCodec(new CuVS2510GPUSearchCodec());
      try (IndexWriter writer = new IndexWriter(directory, config)) {
        for (int i = 0; i < datasetSize; i++) {
          Document doc = new Document();
          doc.add(new KnnFloatVectorField(VECTOR_FIELD, dataset[i], EUCLIDEAN));
          writer.addDocument(doc);
        }
        writer.commit();
      }

      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        assertEquals("expected a single segment", 1, reader.leaves().size());
        LeafReader leaf = reader.leaves().get(0).reader();
        float[][] queries = generateDataset(random(), 32, dimensions);

        // Category c = ordinal % NUM_CATEGORIES; acceptDocs keeps ~1/24 of the ordinals, so the
        // segment's trailing ordinals are rejected. (docId == vector ordinal for a dense segment.)
        for (int c = 0; c < NUM_CATEGORIES; c++) {
          FixedBitSet acceptDocs = new FixedBitSet(datasetSize);
          for (int i = c; i < datasetSize; i += NUM_CATEGORIES) {
            acceptDocs.set(i);
          }
          for (float[] q : queries) {
            TopKnnCollector collector = new TopKnnCollector(topK, Integer.MAX_VALUE);
            leaf.searchNearestVectors(VECTOR_FIELD, q, collector, acceptDocs);
            for (ScoreDoc hit : collector.topDocs().scoreDocs) {
              assertTrue(
                  "per-segment search returned doc " + hit.doc + " outside filter category " + c,
                  acceptDocs.get(hit.doc));
            }
          }
        }
      }
    }
  }
}

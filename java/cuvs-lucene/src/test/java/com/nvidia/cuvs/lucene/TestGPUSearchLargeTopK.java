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
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.Test;

/**
 * Searches a GPU index that has no brute force index, the default, for more hits than {@link
 * CuVS2510GPUVectorsReader#MAX_CAGRA_K}. Since Lucene 10.3, Lucene's own {@link
 * KnnFloatVectorQuery} asks a segment for as many hits as it expects from it, which for a segment
 * holding most of the index exceeds {@code k}, and so can exceed that limit even when {@code k}
 * does not.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestGPUSearchLargeTopK extends LuceneTestCase {

  private static final String VECTOR_FIELD = "vectors";
  private static final int DIMENSIONS = 8;

  @Test
  public void largeSegmentWithLargeTopKIsSearched() throws Exception {
    assumeTrue("cuVS not supported", isSupported());
    // A segment holding 90% of the documents: with k = 1000, Lucene 10.3 asks it for about 1052.
    int k = 1000;
    int[] segmentSizes = {9000, 1000};
    try (Directory dir = newDirectory(new ByteBuffersDirectory())) {
      IndexWriterConfig config =
          new IndexWriterConfig()
              .setCodec(CuVSCodecs.gpuSearch())
              .setMergePolicy(NoMergePolicy.INSTANCE);
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

      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        assertEquals(segmentSizes.length, reader.leaves().size());
        float[] query = generateDataset(random(), 1, DIMENSIONS)[0];
        int hits =
            new IndexSearcher(reader)
                .search(new KnnFloatVectorQuery(VECTOR_FIELD, query, k), k)
                .scoreDocs
                .length;
        assertEquals(k, hits);
      }
    }
  }
}

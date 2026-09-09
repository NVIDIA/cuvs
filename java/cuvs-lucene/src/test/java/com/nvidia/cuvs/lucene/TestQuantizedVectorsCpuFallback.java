/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.getCuVSResourcesInstance;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.setCuVSResourcesInstance;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;
import static org.junit.Assert.assertEquals;

import com.nvidia.cuvs.CuVSResources;
import java.util.Arrays;
import java.util.concurrent.Callable;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.junit.Test;

/** Exercises both quantized codecs through their intentional Lucene CPU-writer fallback. */
public class TestQuantizedVectorsCpuFallback {

  private static final String VECTOR_FIELD = "vector";
  private static final int DIMENSIONS = 128;
  private static final int MAX_CONN = 16;
  private static final int BEAM_WIDTH = 100;

  @Test
  public void testBinaryQuantizedCodecWritesAndSearchesWithCpuFallback() throws Exception {
    assertWritesAndSearchesWithCpuFallback(
        () -> new LuceneAcceleratedHNSWBinaryQuantizedCodec(fallbackParameters()));
  }

  @Test
  public void testScalarQuantizedCodecWritesAndSearchesWithCpuFallback() throws Exception {
    assertWritesAndSearchesWithCpuFallback(
        () -> new LuceneAcceleratedHNSWScalarQuantizedCodec(fallbackParameters()));
  }

  private static AcceleratedHNSWParams fallbackParameters() {
    return new AcceleratedHNSWParams.Builder()
        .withMaxConn(MAX_CONN)
        .withBeamWidth(BEAM_WIDTH)
        .build();
  }

  private static void assertWritesAndSearchesWithCpuFallback(Callable<Codec> codecFactory)
      throws Exception {
    CuVSResources previousResources = getCuVSResourcesInstance();
    try {
      setCuVSResourcesInstance(null);
      try (Directory directory = new ByteBuffersDirectory()) {
        writeTwoVectors(directory, codecFactory.call());
        assertFirstVectorIsItsOwnNearestNeighbor(directory);
      }
    } finally {
      setCuVSResourcesInstance(previousResources);
    }
  }

  private static void writeTwoVectors(Directory directory, Codec codec) throws Exception {
    float[] firstVector = new float[DIMENSIONS];
    float[] secondVector = new float[DIMENSIONS];
    Arrays.fill(secondVector, 1.0f);

    IndexWriterConfig config = new IndexWriterConfig().setCodec(codec).setUseCompoundFile(false);
    try (IndexWriter writer = new IndexWriter(directory, config)) {
      writer.addDocument(vectorDocument(firstVector));
      writer.addDocument(vectorDocument(secondVector));
    }
  }

  private static Document vectorDocument(float[] vector) {
    Document document = new Document();
    document.add(new KnnFloatVectorField(VECTOR_FIELD, vector, EUCLIDEAN));
    return document;
  }

  private static void assertFirstVectorIsItsOwnNearestNeighbor(Directory directory)
      throws Exception {
    try (DirectoryReader reader = DirectoryReader.open(directory)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      KnnFloatVectorQuery query = new KnnFloatVectorQuery(VECTOR_FIELD, new float[DIMENSIONS], 1);

      var hits = searcher.search(query, 1).scoreDocs;

      assertEquals(1, hits.length);
      assertEquals(0, hits[0].doc);
    }
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.FLAT_LAYOUT_ATTRIBUTE_KEY;
import static com.nvidia.cuvs.lucene.LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.LUCENE_102_BINARY_FLAT_LAYOUT;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.getCuVSResourcesInstance;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.setCuVSResourcesInstance;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import com.nvidia.cuvs.CuVSResources;
import java.util.Arrays;
import java.util.concurrent.Callable;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.junit.Test;

/** Exercises quantized codecs through their intentional Lucene CPU-writer fallback. */
public class TestQuantizedVectorsCpuFallback {

  private static final String VECTOR_FIELD = "vector";
  private static final int DIMENSIONS = 128;
  private static final int MAX_CONN = 17;
  private static final int BEAM_WIDTH = 101;

  @Test
  public void testBinaryQuantizedCodecWritesAndSearchesWithCpuFallback() throws Exception {
    withCuvsDisabled(
        () -> {
          try (Directory directory = new ByteBuffersDirectory()) {
            writeTwoVectors(
                directory,
                new LuceneAcceleratedHNSWBinaryQuantizedCodec(fallbackParameters()),
                false);
            assertBinaryFallbackIsMarked(directory);
            assertFirstVectorIsItsOwnNearestNeighbor(directory);
            assertPersistedMaxConn(directory);
            assertTrue(
                "CPU fallback did not write Lucene102 binary metadata",
                Arrays.stream(directory.listAll()).anyMatch(file -> file.endsWith(".vemb")));
          }
          return null;
        });
  }

  @Test
  public void testBinaryQuantizedCodecReadsCompoundCpuFallback() throws Exception {
    withCuvsDisabled(
        () -> {
          try (Directory directory = new ByteBuffersDirectory()) {
            writeTwoVectors(
                directory,
                new LuceneAcceleratedHNSWBinaryQuantizedCodec(fallbackParameters()),
                true);
            assertSingleCompoundSegment(directory);
            assertBinaryFallbackIsMarked(directory);
            assertFirstVectorIsItsOwnNearestNeighbor(directory);
          }
          return null;
        });
  }

  @Test
  public void testScalarQuantizedCodecWritesAndSearchesWithCpuFallback() throws Exception {
    withCuvsDisabled(
        () -> {
          try (Directory directory = new ByteBuffersDirectory()) {
            writeTwoVectors(
                directory,
                new LuceneAcceleratedHNSWScalarQuantizedCodec(fallbackParameters()),
                false);
            assertFirstVectorIsItsOwnNearestNeighbor(directory);
            assertPersistedMaxConn(directory);
            assertTrue(
                "CPU fallback did not write scalar-quantized metadata",
                Arrays.stream(directory.listAll()).anyMatch(file -> file.endsWith(".vemq")));
            assertTrue(
                "CPU fallback did not write scalar-quantized vectors",
                Arrays.stream(directory.listAll()).anyMatch(file -> file.endsWith(".veq")));
          }
          return null;
        });
  }

  @Test
  public void testBinaryCpuFallbackEnforcesLuceneDimensionLimit() throws Exception {
    withCuvsDisabled(
        () -> {
          assertCpuFallbackDimensionLimit(
              new LuceneAcceleratedHNSWBinaryQuantizedCodec(fallbackParameters()));
          return null;
        });
  }

  @Test
  public void testScalarCpuFallbackEnforcesLuceneDimensionLimit() throws Exception {
    withCuvsDisabled(
        () -> {
          assertCpuFallbackDimensionLimit(
              new LuceneAcceleratedHNSWScalarQuantizedCodec(fallbackParameters()));
          return null;
        });
  }

  private static AcceleratedHNSWParams fallbackParameters() {
    return new AcceleratedHNSWParams.Builder()
        .withMaxConn(MAX_CONN)
        .withBeamWidth(BEAM_WIDTH)
        .build();
  }

  private static <T> T withCuvsDisabled(Callable<T> operation) throws Exception {
    CuVSResources previousResources = getCuVSResourcesInstance();
    try {
      setCuVSResourcesInstance(null);
      return operation.call();
    } finally {
      setCuVSResourcesInstance(previousResources);
    }
  }

  private static void writeTwoVectors(Directory directory, Codec codec, boolean useCompoundFile)
      throws Exception {
    float[] firstVector = new float[DIMENSIONS];
    float[] secondVector = new float[DIMENSIONS];
    Arrays.fill(secondVector, 1.0f);

    IndexWriterConfig config =
        new IndexWriterConfig().setCodec(codec).setUseCompoundFile(useCompoundFile);
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

  private static void assertCpuFallbackDimensionLimit(Codec codec) throws Exception {
    int supportedDimensions = KnnVectorsFormat.DEFAULT_MAX_DIMENSIONS;
    assertEquals(supportedDimensions, codec.knnVectorsFormat().getMaxDimensions(VECTOR_FIELD));

    try (Directory directory = new ByteBuffersDirectory()) {
      writeSingleVector(directory, codec, supportedDimensions);
    }

    try (Directory directory = new ByteBuffersDirectory()) {
      int unsupportedDimensions = supportedDimensions + 1;
      IllegalArgumentException error =
          assertThrows(
              IllegalArgumentException.class,
              () -> writeSingleVector(directory, codec, unsupportedDimensions));
      assertTrue(error.getMessage().contains("<= [" + supportedDimensions + "]"));
      assertTrue(error.getMessage().contains("got " + unsupportedDimensions));
    }
  }

  private static void writeSingleVector(Directory directory, Codec codec, int dimensions)
      throws Exception {
    float[] vector = new float[dimensions];
    vector[0] = 1.0f;
    try (IndexWriter writer = new IndexWriter(directory, new IndexWriterConfig().setCodec(codec))) {
      writer.addDocument(vectorDocument(vector));
    }
  }

  private static void assertBinaryFallbackIsMarked(Directory directory) throws Exception {
    SegmentInfos segments = SegmentInfos.readLatestCommit(directory);
    assertEquals(1, segments.size());
    assertEquals(
        LUCENE_102_BINARY_FLAT_LAYOUT,
        segments.info(0).info.getAttribute(FLAT_LAYOUT_ATTRIBUTE_KEY));
  }

  private static void assertSingleCompoundSegment(Directory directory) throws Exception {
    SegmentInfos segments = SegmentInfos.readLatestCommit(directory);
    assertEquals(1, segments.size());
    assertTrue("CPU fallback segment is not compound", segments.info(0).info.getUseCompoundFile());
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

  private static void assertPersistedMaxConn(Directory directory) throws Exception {
    try (DirectoryReader reader = DirectoryReader.open(directory)) {
      KnnVectorsReader vectorsReader =
          ((CodecReader) reader.leaves().getFirst().reader()).getVectorReader();
      if (vectorsReader instanceof PerFieldKnnVectorsFormat.FieldsReader fieldsReader) {
        vectorsReader = fieldsReader.getFieldReader(VECTOR_FIELD);
      }
      assertTrue(vectorsReader instanceof HnswGraphProvider);
      assertEquals(MAX_CONN, ((HnswGraphProvider) vectorsReader).getGraph(VECTOR_FIELD).maxConn());
    }
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.FLAT_LAYOUT_ATTRIBUTE_KEY;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.Map;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.junit.Assume;
import org.junit.Test;

/** Protects quantized GPU limits and the released binary codec's file layout. */
public class TestBinaryQuantizedFormatCompatibility {

  private static final String VECTOR_FIELD = "vector";
  private static final int DOCUMENT_COUNT = 128;
  private static final int GPU_MAX_DIMENSIONS = 4096;

  @Test
  public void testGpuBinaryCodecRetainsLucene99LayoutAtDimensionLimit() throws Exception {
    Assume.assumeTrue("cuVS is not supported", isSupported());
    var format = new LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat(gpuParameters());
    assertEquals(GPU_MAX_DIMENSIONS, format.getMaxDimensions(VECTOR_FIELD));

    try (Directory directory = new ByteBuffersDirectory()) {
      writeVectors(directory, format);

      String[] files = directory.listAll();
      assertTrue(
          "Missing Lucene99 flat metadata: " + Arrays.toString(files), hasFile(files, ".vemf"));
      assertTrue(
          "Missing Lucene99 flat vectors: " + Arrays.toString(files), hasFile(files, ".vec"));
      assertFalse(
          "GPU layout unexpectedly contains Lucene102 binary metadata: " + Arrays.toString(files),
          hasFile(files, ".vemb"));
      assertFalse(
          "GPU layout unexpectedly contains Lucene102 binary vectors: " + Arrays.toString(files),
          hasFile(files, ".veb"));
      assertNull(
          "released GPU layout must remain unmarked",
          SegmentInfos.readLatestCommit(directory)
              .info(0)
              .info
              .getAttribute(FLAT_LAYOUT_ATTRIBUTE_KEY));

      assertVectorsAreReadableAndSearchable(directory);
    }
  }

  @Test
  public void testGpuScalarCodecWritesAndSearchesAtDimensionLimit() throws Exception {
    Assume.assumeTrue("cuVS is not supported", isSupported());
    var format = new LuceneAcceleratedHNSWScalarQuantizedVectorsFormat(gpuParameters());
    assertEquals(GPU_MAX_DIMENSIONS, format.getMaxDimensions(VECTOR_FIELD));

    try (Directory directory = new ByteBuffersDirectory()) {
      writeVectors(directory, format);
      assertVectorsAreReadableAndSearchable(directory);
    }
  }

  @Test
  public void testConflictingLayoutMarkerIsRejectedWithoutBeingOverwritten() throws Exception {
    try (Directory directory = new ByteBuffersDirectory()) {
      var segment =
          new SegmentInfo(
              directory,
              Version.LATEST,
              Version.LATEST,
              "_conflict",
              0,
              false,
              false,
              Codec.getDefault(),
              Map.of(),
              StringHelper.randomId(),
              Map.of(),
              null);
      String incompatibleLayout = "future-layout";
      segment.putAttribute(FLAT_LAYOUT_ATTRIBUTE_KEY, incompatibleLayout);

      IllegalStateException error =
          assertThrows(
              IllegalStateException.class,
              () ->
                  LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.markLucene102BinaryLayout(
                      segment));

      assertTrue(error.getMessage().contains(incompatibleLayout));
      assertEquals(incompatibleLayout, segment.getAttribute(FLAT_LAYOUT_ATTRIBUTE_KEY));
    }
  }

  private static Document vectorDocument(int document) {
    float[] vector = vectorForDocument(document);
    Document result = new Document();
    result.add(new KnnFloatVectorField(VECTOR_FIELD, vector, EUCLIDEAN));
    return result;
  }

  private static float[] vectorForDocument(int document) {
    float[] vector = new float[GPU_MAX_DIMENSIONS];
    vector[document % GPU_MAX_DIMENSIONS] = 1.0f;
    vector[(document * 7 + 3) % GPU_MAX_DIMENSIONS] += 0.5f;
    return vector;
  }

  private static AcceleratedHNSWParams gpuParameters() {
    return new AcceleratedHNSWParams.Builder()
        .withStrategy(AcceleratedHNSWParams.Strategy.CUSTOM)
        .withGraphDegree(32)
        .withIntermediateGraphDegree(64)
        .withHNSWLayer(1)
        .build();
  }

  private static void writeVectors(Directory directory, KnnVectorsFormat format) throws Exception {
    IndexWriterConfig config =
        new IndexWriterConfig()
            .setUseCompoundFile(false)
            .setCodec(TestUtil.alwaysKnnVectorsFormat(format));
    try (IndexWriter writer = new IndexWriter(directory, config)) {
      for (int document = 0; document < DOCUMENT_COUNT; document++) {
        writer.addDocument(vectorDocument(document));
      }
    }
  }

  private static void assertVectorsAreReadableAndSearchable(Directory directory) throws Exception {
    try (DirectoryReader reader = DirectoryReader.open(directory)) {
      var values = reader.leaves().getFirst().reader().getFloatVectorValues(VECTOR_FIELD);
      assertNotNull(values);
      assertEquals(DOCUMENT_COUNT, values.size());
      assertEquals(GPU_MAX_DIMENSIONS, values.dimension());

      var hits =
          new IndexSearcher(reader)
              .search(new KnnFloatVectorQuery(VECTOR_FIELD, vectorForDocument(0), 1), 1)
              .scoreDocs;
      assertEquals(1, hits.length);
      assertEquals("the query document must rank first", 0, hits[0].doc);
    }
  }

  private static boolean hasFile(String[] files, String extension) {
    return Arrays.stream(files).anyMatch(file -> file.endsWith(extension));
  }
}

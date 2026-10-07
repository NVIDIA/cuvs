/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.FLAT_LAYOUT_ATTRIBUTE_KEY;
import static com.nvidia.cuvs.lucene.LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.LUCENE_102_BINARY_FLAT_LAYOUT;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.getCuVSResourcesInstance;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.setCuVSResourcesInstance;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import com.nvidia.cuvs.CuVSResources;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.Base64;
import java.util.HexFormat;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.concurrent.Callable;
import java.util.function.IntFunction;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.junit.Assume;
import org.junit.Test;

/** Protects the binary accelerated codec's persisted layout across cuvs-lucene releases. */
public class TestBinaryQuantizedFormatCompatibility {

  private static final String VECTOR_FIELD = "vector";
  private static final String STORED_ID_FIELD = "id";
  private static final int DOCUMENT_COUNT = 128;
  // With M=16, this yields about 130 nodes at level 2, safely above the CAGRA intermediate degree.
  private static final int MULTI_LAYER_DOCUMENT_COUNT = 33_280;
  private static final int DIMENSIONS = 128;
  private static final int UNALIGNED_DIMENSIONS = 130;
  private static final int GPU_MAX_DIMENSIONS = 4096;
  private static final int SEARCH_CANDIDATES = 10;
  private static final double MINIMUM_RECALL = 0.8;
  // The Lucene102 CPU baseline and pre-change one-layer GPU control both reach 4/10 on the
  // hardest mixed-fixture query.
  private static final double MINIMUM_MIXED_LAYOUT_RECALL = 0.4;
  private static final String RELEASED_FIXTURE_RESOURCE =
      "/com/nvidia/cuvs/lucene/cuvs-lucene-26.08.0-binary-index.zip.b64";
  private static final String RELEASED_FIXTURE_SHA256 =
      "157c4711b083d1985f78424e1543008abe8111fdb887d83369be983d6bd06f69";

  @Test
  public void testReadsAndSearchesReleasedUnmarkedLucene99Index() throws Exception {
    try (Directory directory = loadReleasedFixture()) {
      SegmentInfo releasedSegment = onlySegment(directory);

      assertNull(releasedSegment.getAttribute(FLAT_LAYOUT_ATTRIBUTE_KEY));
      assertTrue(hasFile(directory, ".vemf"));
      assertTrue(hasFile(directory, ".vec"));
      assertFalse(hasFile(directory, ".vemb"));
      assertFalse(hasFile(directory, ".veb"));
      assertVectorCount(directory, DOCUMENT_COUNT);
      assertNearestStoredId(directory, releasedVector(37), 37);
    }
  }

  @Test
  public void testGpuWriterMarksAndSearchesNonCompoundLucene102BinaryLayout() throws Exception {
    Assume.assumeTrue("cuVS is not supported", isSupported());

    try (Directory directory = new ByteBuffersDirectory()) {
      writeNewSegment(directory, false, 0, GPU_MAX_DIMENSIONS);

      assertEquals(
          LUCENE_102_BINARY_FLAT_LAYOUT,
          onlySegment(directory).getAttribute(FLAT_LAYOUT_ATTRIBUTE_KEY));
      assertTrue(hasFile(directory, ".vemf"));
      assertTrue(hasFile(directory, ".vec"));
      assertTrue(hasFile(directory, ".vemb"));
      assertTrue(hasFile(directory, ".veb"));
      assertVectorCountAndDimensions(directory, DOCUMENT_COUNT, GPU_MAX_DIMENSIONS);
      assertNearestStoredId(directory, newVector(19, GPU_MAX_DIMENSIONS), 19);
    }
  }

  @Test
  public void testGpuScalarCodecWritesAndSearchesAtDimensionLimit() throws Exception {
    Assume.assumeTrue("cuVS is not supported", isSupported());

    try (Directory directory = new ByteBuffersDirectory()) {
      writeScalarSegment(directory, GPU_MAX_DIMENSIONS);

      assertVectorCountAndDimensions(directory, DOCUMENT_COUNT, GPU_MAX_DIMENSIONS);
      assertNearestStoredId(directory, newVector(41, GPU_MAX_DIMENSIONS), 41);
    }
  }

  @Test
  public void testGpuWriterMarksAndSearchesCompoundLucene102BinaryLayout() throws Exception {
    Assume.assumeTrue("cuVS is not supported", isSupported());

    try (Directory directory = new ByteBuffersDirectory()) {
      writeNewSegment(directory, true, 0);

      SegmentInfo segment = onlySegment(directory);
      assertTrue("GPU binary segment is not compound", segment.getUseCompoundFile());
      assertEquals(LUCENE_102_BINARY_FLAT_LAYOUT, segment.getAttribute(FLAT_LAYOUT_ATTRIBUTE_KEY));
      assertVectorCount(directory, DOCUMENT_COUNT);
      assertNearestStoredId(directory, newVector(73), 73);
    }
  }

  @Test
  public void testGpuWriterBuildsAndSearchesThreeLayerGraphFromOriginalFloatVectors()
      throws Exception {
    Assume.assumeTrue("cuVS is not supported", isSupported());

    try (Directory directory = new ByteBuffersDirectory()) {
      writeMultiLayerSegment(directory);

      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        assertEquals(3, graphForVectorField(reader).numLevels());
      }
      for (int queryId :
          new int[] {0, MULTI_LAYER_DOCUMENT_COUNT / 2, MULTI_LAYER_DOCUMENT_COUNT - 1}) {
        assertRecallAgainstBruteForce(
            directory,
            queryId,
            MULTI_LAYER_DOCUMENT_COUNT,
            id -> newVector(id, DIMENSIONS),
            "three-layer binary graph",
            MINIMUM_RECALL);
      }
    }
  }

  @Test
  public void testGpuWriterSearchesUnalignedDimensions() throws Exception {
    Assume.assumeTrue("cuVS is not supported", isSupported());

    try (Directory directory = new ByteBuffersDirectory()) {
      writeNewSegment(directory, false, 0, UNALIGNED_DIMENSIONS);

      assertEquals(
          LUCENE_102_BINARY_FLAT_LAYOUT,
          onlySegment(directory).getAttribute(FLAT_LAYOUT_ATTRIBUTE_KEY));
      assertVectorCount(directory, DOCUMENT_COUNT);
      assertNearestStoredId(directory, newVector(29, UNALIGNED_DIMENSIONS), 29);
    }
  }

  @Test
  public void testGpuWriterRecallTracksBruteForceAtAlignedAndUnalignedDimensions()
      throws Exception {
    Assume.assumeTrue("cuVS is not supported", isSupported());

    for (int dimensions : new int[] {DIMENSIONS, UNALIGNED_DIMENSIONS}) {
      try (Directory directory = new ByteBuffersDirectory()) {
        writeNewSegment(directory, false, 0, dimensions);
        for (int queryId : new int[] {0, 17, 63, 127}) {
          assertRecallAgainstBruteForce(directory, queryId, dimensions);
        }
      }
    }
  }

  @Test
  public void testCpuFallbackMergeRewritesMixedSegmentsToMarkedLayout() throws Exception {
    withCuvsDisabled(() -> verifyMixedSegmentsRewriteToMarkedLayout());
  }

  @Test
  public void testGpuMergeRewritesMixedSegmentsToMarkedLayout() throws Exception {
    Assume.assumeTrue("cuVS is not supported", isSupported());

    verifyMixedSegmentsRewriteToMarkedLayout();
  }

  @Test
  public void testUnknownLayoutMarkerFailsWithSegmentContext() throws Exception {
    withCuvsDisabled(
        () -> {
          try (Directory directory = new ByteBuffersDirectory()) {
            writeNewSegment(directory, false, 0);
            SegmentInfo segment = onlySegment(directory);
            segment.putAttribute(FLAT_LAYOUT_ATTRIBUTE_KEY, "future-layout");

            CorruptIndexException failure =
                assertThrows(
                    CorruptIndexException.class,
                    () -> binaryFormat().fieldsReader(readState(directory, segment)));

            assertTrue(failure.getMessage().contains("Unsupported binary flat-vector layout"));
            assertTrue(failure.getMessage().contains("future-layout"));
            assertTrue(failure.getMessage().contains(segment.name));
          }
          return null;
        });
  }

  @Test
  public void testUnmarkedLucene102FilesFailInsteadOfGuessing() throws Exception {
    withCuvsDisabled(
        () -> {
          try (Directory directory = new ByteBuffersDirectory()) {
            writeNewSegment(directory, false, 0);
            SegmentInfo segment = onlySegment(directory);
            segment.putAttribute(FLAT_LAYOUT_ATTRIBUTE_KEY, null);

            CorruptIndexException failure =
                assertThrows(
                    CorruptIndexException.class,
                    () -> binaryFormat().fieldsReader(readState(directory, segment)));

            assertTrue(failure.getMessage().contains("without the required layout marker"));
            assertTrue(failure.getMessage().contains(FLAT_LAYOUT_ATTRIBUTE_KEY));
          }
          return null;
        });
  }

  @Test
  public void testIncompleteLucene102FilesFailBeforeOpeningReader() throws Exception {
    withCuvsDisabled(
        () -> {
          try (Directory directory = new ByteBuffersDirectory()) {
            writeNewSegment(directory, false, 0);
            SegmentInfo segment = onlySegment(directory);
            directory.deleteFile(findFile(directory, ".veb"));

            CorruptIndexException failure =
                assertThrows(
                    CorruptIndexException.class,
                    () -> binaryFormat().fieldsReader(readState(directory, segment)));

            assertTrue(failure.getMessage().contains("files are incomplete"));
            assertTrue(failure.getMessage().contains("metadata=true, data=false"));
          }
          return null;
        });
  }

  @Test
  public void testMarkerWithoutLucene102FilesFailsInsteadOfReadingLegacyLayout() throws Exception {
    try (Directory directory = loadReleasedFixture()) {
      SegmentInfo segment = onlySegment(directory);
      segment.putAttribute(FLAT_LAYOUT_ATTRIBUTE_KEY, LUCENE_102_BINARY_FLAT_LAYOUT);

      CorruptIndexException failure =
          assertThrows(
              CorruptIndexException.class,
              () -> binaryFormat().fieldsReader(readState(directory, segment)));

      assertTrue(failure.getMessage().contains("binary flat vectors, but their files are absent"));
      assertTrue(failure.getMessage().contains(segment.name));
    }
  }

  @Test
  public void testConflictingLayoutMarkerIsRejectedWithoutBeingOverwritten() throws Exception {
    withCuvsDisabled(
        () -> {
          try (Directory directory = new ByteBuffersDirectory()) {
            writeNewSegment(directory, false, 0);
            SegmentInfo segment = onlySegment(directory);
            String incompatibleLayout = "future-layout";
            segment.putAttribute(FLAT_LAYOUT_ATTRIBUTE_KEY, incompatibleLayout);

            IllegalStateException failure =
                assertThrows(
                    IllegalStateException.class,
                    () ->
                        LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.markLucene102BinaryLayout(
                            segment));

            assertTrue(failure.getMessage().contains(incompatibleLayout));
            assertEquals(incompatibleLayout, segment.getAttribute(FLAT_LAYOUT_ATTRIBUTE_KEY));
          }
          return null;
        });
  }

  private static void appendNewSegmentWithoutMerging(Directory directory) throws Exception {
    IndexWriterConfig config =
        writerConfig(false)
            .setOpenMode(IndexWriterConfig.OpenMode.APPEND)
            .setMergePolicy(NoMergePolicy.INSTANCE);
    try (IndexWriter writer = new IndexWriter(directory, config)) {
      for (int id = DOCUMENT_COUNT; id < DOCUMENT_COUNT * 2; id++) {
        writer.addDocument(vectorDocument(id, currentVector(id)));
      }
    }
  }

  private static void forceMerge(Directory directory) throws Exception {
    IndexWriterConfig config = writerConfig(false).setOpenMode(IndexWriterConfig.OpenMode.APPEND);
    try (IndexWriter writer = new IndexWriter(directory, config)) {
      writer.forceMerge(1);
    }
  }

  private static Void verifyMixedSegmentsRewriteToMarkedLayout() throws Exception {
    try (Directory directory = loadReleasedFixture()) {
      appendNewSegmentWithoutMerging(directory);

      SegmentInfos mixedSegments = SegmentInfos.readLatestCommit(directory);
      assertEquals(2, mixedSegments.size());
      assertEquals(1, countLayout(mixedSegments, null));
      assertEquals(1, countLayout(mixedSegments, LUCENE_102_BINARY_FLAT_LAYOUT));
      assertNearestStoredId(directory, releasedVector(31), 31);
      assertNearestStoredId(directory, currentVector(159), 159);

      forceMerge(directory);

      SegmentInfos mergedSegments = SegmentInfos.readLatestCommit(directory);
      assertEquals(1, mergedSegments.size());
      assertEquals(
          LUCENE_102_BINARY_FLAT_LAYOUT,
          mergedSegments.info(0).info.getAttribute(FLAT_LAYOUT_ATTRIBUTE_KEY));
      assertVectorCount(directory, DOCUMENT_COUNT * 2);
      for (int queryId : new int[] {0, 31, 128, 159, 255}) {
        assertRecallAgainstBruteForce(
            directory,
            queryId,
            DOCUMENT_COUNT * 2,
            TestBinaryQuantizedFormatCompatibility::mixedVector,
            "merged released/current graph",
            MINIMUM_MIXED_LAYOUT_RECALL);
      }
    }
    return null;
  }

  private static void writeNewSegment(Directory directory, boolean useCompoundFile, int firstId)
      throws Exception {
    writeNewSegment(directory, useCompoundFile, firstId, DIMENSIONS);
  }

  private static void writeNewSegment(
      Directory directory, boolean useCompoundFile, int firstId, int dimensions) throws Exception {
    IndexWriterConfig config = writerConfig(useCompoundFile);
    try (IndexWriter writer = new IndexWriter(directory, config)) {
      addDocuments(writer, firstId, DOCUMENT_COUNT, dimensions);
    }
  }

  private static void writeMultiLayerSegment(Directory directory) throws Exception {
    IndexWriterConfig config = writerConfig(false, threeLayerBinaryParameters());
    config.setMaxBufferedDocs(MULTI_LAYER_DOCUMENT_COUNT + 1);
    config.setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);
    try (IndexWriter writer = new IndexWriter(directory, config)) {
      addDocuments(writer, 0, MULTI_LAYER_DOCUMENT_COUNT, DIMENSIONS);
    }
  }

  private static void writeScalarSegment(Directory directory, int dimensions) throws Exception {
    IndexWriterConfig config =
        new IndexWriterConfig()
            .setUseCompoundFile(false)
            .setCodec(new LuceneAcceleratedHNSWScalarQuantizedCodec(binaryParameters()));
    try (IndexWriter writer = new IndexWriter(directory, config)) {
      addDocuments(writer, 0, DOCUMENT_COUNT, dimensions);
    }
  }

  private static IndexWriterConfig writerConfig(boolean useCompoundFile) throws Exception {
    return writerConfig(useCompoundFile, binaryParameters());
  }

  private static IndexWriterConfig writerConfig(
      boolean useCompoundFile, AcceleratedHNSWParams parameters) throws Exception {
    return new IndexWriterConfig()
        .setUseCompoundFile(useCompoundFile)
        .setCodec(new LuceneAcceleratedHNSWBinaryQuantizedCodec(parameters));
  }

  private static LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat binaryFormat() {
    return new LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat(binaryParameters());
  }

  private static AcceleratedHNSWParams binaryParameters() {
    return new AcceleratedHNSWParams.Builder()
        .withStrategy(AcceleratedHNSWParams.Strategy.CUSTOM)
        .withGraphDegree(32)
        .withIntermediateGraphDegree(64)
        .withHNSWLayer(1)
        .withMaxConn(32)
        .withBeamWidth(100)
        .build();
  }

  private static AcceleratedHNSWParams threeLayerBinaryParameters() {
    return new AcceleratedHNSWParams.Builder()
        .withStrategy(AcceleratedHNSWParams.Strategy.CUSTOM)
        .withGraphDegree(32)
        .withIntermediateGraphDegree(64)
        .withHNSWLayer(3)
        .withMaxConn(32)
        .withBeamWidth(100)
        .build();
  }

  private static void addDocuments(IndexWriter writer, int firstId, int count) throws IOException {
    addDocuments(writer, firstId, count, DIMENSIONS);
  }

  private static void addDocuments(IndexWriter writer, int firstId, int count, int dimensions)
      throws IOException {
    for (int id = firstId; id < firstId + count; id++) {
      writer.addDocument(vectorDocument(id, newVector(id, dimensions)));
    }
  }

  private static Document vectorDocument(int id, float[] vector) {
    Document document = new Document();
    document.add(new StoredField(STORED_ID_FIELD, id));
    document.add(new KnnFloatVectorField(VECTOR_FIELD, vector, EUCLIDEAN));
    return document;
  }

  private static float[] releasedVector(int id) {
    float[] vector = new float[DIMENSIONS];
    vector[id % DIMENSIONS] = 1.0f;
    vector[(id * 7 + 3) % DIMENSIONS] += 0.5f;
    return vector;
  }

  private static float[] newVector(int id) {
    return newVector(id, DIMENSIONS);
  }

  private static float[] newVector(int id, int dimensions) {
    SplittableRandom random = new SplittableRandom(0x5EEDB1A4L + id);
    float[] vector = new float[dimensions];
    for (int dimension = 0; dimension < dimensions; dimension++) {
      vector[dimension] = random.nextBoolean() ? 1.0f : -1.0f;
    }
    return vector;
  }

  private static float[] currentVector(int id) {
    float[] vector = newVector(id);
    // Keep the dense vectors' squared norm (1.28) near the sparse fixture vectors' 1.25.
    for (int dimension = 0; dimension < vector.length; dimension++) {
      vector[dimension] *= 0.1f;
    }
    return vector;
  }

  private static float[] mixedVector(int id) {
    return id < DOCUMENT_COUNT ? releasedVector(id) : currentVector(id);
  }

  private static void assertVectorCount(Directory directory, int expected) throws IOException {
    try (DirectoryReader reader = DirectoryReader.open(directory)) {
      int actual = 0;
      for (var context : reader.leaves()) {
        var values = context.reader().getFloatVectorValues(VECTOR_FIELD);
        if (values != null) {
          actual += values.size();
        }
      }
      assertEquals(expected, actual);
    }
  }

  private static void assertVectorCountAndDimensions(
      Directory directory, int expectedCount, int expectedDimensions) throws IOException {
    try (DirectoryReader reader = DirectoryReader.open(directory)) {
      int actualCount = 0;
      for (var context : reader.leaves()) {
        var values = context.reader().getFloatVectorValues(VECTOR_FIELD);
        if (values != null) {
          actualCount += values.size();
          assertEquals(expectedDimensions, values.dimension());
        }
      }
      assertEquals(expectedCount, actualCount);
    }
  }

  private static void assertNearestStoredId(Directory directory, float[] query, int expectedId)
      throws IOException {
    try (DirectoryReader reader = DirectoryReader.open(directory)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      var hits =
          searcher.search(
                  new KnnFloatVectorQuery(VECTOR_FIELD, query, SEARCH_CANDIDATES),
                  SEARCH_CANDIDATES)
              .scoreDocs;

      assertEquals(SEARCH_CANDIDATES, hits.length);
      Number storedId =
          searcher.storedFields().document(hits[0].doc).getField(STORED_ID_FIELD).numericValue();
      assertEquals(expectedId, storedId.intValue());
    }
  }

  private static void assertRecallAgainstBruteForce(
      Directory directory, int queryId, int dimensions) throws IOException {
    assertRecallAgainstBruteForce(
        directory,
        queryId,
        DOCUMENT_COUNT,
        id -> newVector(id, dimensions),
        "dimensions=" + dimensions,
        MINIMUM_RECALL);
  }

  private static void assertRecallAgainstBruteForce(
      Directory directory,
      int queryId,
      int documentCount,
      IntFunction<float[]> vectorForId,
      String context,
      double minimumRecall)
      throws IOException {
    float[] query = vectorForId.apply(queryId);
    double[] distances = new double[documentCount];
    for (int id = 0; id < documentCount; id++) {
      distances[id] = squaredEuclideanDistance(query, vectorForId.apply(id));
    }
    double[] sortedDistances = distances.clone();
    Arrays.sort(sortedDistances);
    double topKDistance = sortedDistances[SEARCH_CANDIDATES - 1];

    try (DirectoryReader reader = DirectoryReader.open(directory)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      var hits =
          searcher.search(
                  new KnnFloatVectorQuery(VECTOR_FIELD, query, SEARCH_CANDIDATES),
                  SEARCH_CANDIDATES)
              .scoreDocs;

      assertEquals(SEARCH_CANDIDATES, hits.length);
      Set<Integer> actualIds = new LinkedHashSet<>();
      int relevantHits = 0;
      for (var hit : hits) {
        int id =
            searcher
                .storedFields()
                .document(hit.doc)
                .getField(STORED_ID_FIELD)
                .numericValue()
                .intValue();
        assertTrue("duplicate result for stored id " + id, actualIds.add(id));
        if (distances[id] <= topKDistance) {
          relevantHits++;
        }
      }
      assertEquals(
          "the query document must rank first; returned ids=" + actualIds,
          queryId,
          hitsStoredId(searcher, hits[0]));
      assertTrue(
          "recall fell below "
              + minimumRecall
              + " for "
              + context
              + ", queryId="
              + queryId
              + ", relevantHits="
              + relevantHits,
          relevantHits >= Math.ceil(minimumRecall * SEARCH_CANDIDATES));
    }
  }

  private static HnswGraph graphForVectorField(DirectoryReader reader) throws IOException {
    assertEquals(1, reader.leaves().size());
    KnnVectorsReader vectorReader =
        ((CodecReader) reader.leaves().get(0).reader()).getVectorReader();
    if (vectorReader instanceof PerFieldKnnVectorsFormat.FieldsReader fieldsReader) {
      vectorReader = fieldsReader.getFieldReader(VECTOR_FIELD);
    }
    return ((HnswGraphProvider) vectorReader).getGraph(VECTOR_FIELD);
  }

  private static int hitsStoredId(IndexSearcher searcher, org.apache.lucene.search.ScoreDoc hit)
      throws IOException {
    return searcher
        .storedFields()
        .document(hit.doc)
        .getField(STORED_ID_FIELD)
        .numericValue()
        .intValue();
  }

  private static double squaredEuclideanDistance(float[] left, float[] right) {
    double distance = 0;
    for (int dimension = 0; dimension < left.length; dimension++) {
      double delta = left[dimension] - right[dimension];
      distance += delta * delta;
    }
    return distance;
  }

  private static int countLayout(SegmentInfos segments, String expectedLayout) {
    int matches = 0;
    for (var segment : segments) {
      if (java.util.Objects.equals(
          expectedLayout, segment.info.getAttribute(FLAT_LAYOUT_ATTRIBUTE_KEY))) {
        matches++;
      }
    }
    return matches;
  }

  private static SegmentInfo onlySegment(Directory directory) throws IOException {
    SegmentInfos segments = SegmentInfos.readLatestCommit(directory);
    assertEquals(1, segments.size());
    return segments.info(0).info;
  }

  private static SegmentReadState readState(Directory directory, SegmentInfo segment) {
    return new SegmentReadState(directory, segment, FieldInfos.EMPTY, IOContext.DEFAULT);
  }

  private static boolean hasFile(Directory directory, String extension) throws IOException {
    return Arrays.stream(directory.listAll()).anyMatch(file -> file.endsWith(extension));
  }

  private static String findFile(Directory directory, String extension) throws IOException {
    return Arrays.stream(directory.listAll())
        .filter(file -> file.endsWith(extension))
        .findFirst()
        .orElseThrow(() -> new AssertionError("Missing index file with extension " + extension));
  }

  private static Directory loadReleasedFixture() throws Exception {
    byte[] zipBytes;
    try (InputStream encoded =
        TestBinaryQuantizedFormatCompatibility.class.getResourceAsStream(
            RELEASED_FIXTURE_RESOURCE)) {
      assertNotNull("Missing released-index fixture", encoded);
      zipBytes =
          Base64.getMimeDecoder()
              .decode(new String(encoded.readAllBytes(), StandardCharsets.US_ASCII));
    }
    assertEquals(
        RELEASED_FIXTURE_SHA256,
        HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(zipBytes)));

    Directory directory = new ByteBuffersDirectory();
    boolean success = false;
    try (ZipInputStream zip = new ZipInputStream(new java.io.ByteArrayInputStream(zipBytes))) {
      for (ZipEntry entry = zip.getNextEntry(); entry != null; entry = zip.getNextEntry()) {
        if (entry.isDirectory() || entry.getName().startsWith("META-INF/")) {
          continue;
        }
        byte[] contents = zip.readAllBytes();
        try (IndexOutput output = directory.createOutput(entry.getName(), IOContext.DEFAULT)) {
          output.writeBytes(contents, contents.length);
        }
      }
      success = true;
      return directory;
    } finally {
      if (success == false) {
        directory.close();
      }
    }
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
}

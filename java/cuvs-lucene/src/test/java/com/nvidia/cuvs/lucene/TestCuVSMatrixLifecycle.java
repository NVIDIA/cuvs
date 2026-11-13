/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.closeCuVSResourcesInstance;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.setCuVSResourcesInstance;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.spi.CuVSProvider;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.HashSet;
import java.util.List;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.store.MockDirectoryWrapper;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.Test;

@SuppressSysoutChecks(bugUrl = "")
public class TestCuVSMatrixLifecycle extends LuceneTestCase {

  private static final String VECTOR_FIELD = "vector";
  private static final int DOCUMENT_COUNT = 1024;
  private static final int DIMENSIONS = 128;
  private static final int QUERY_DOCUMENT = 137;
  private static final int TOP_K = 10;
  private static final int QUERY_REPETITIONS = 128;

  @Test
  public void testInvalidFloatVectorReleasesPartiallyBuiltMatrix() throws Throwable {
    assertNoWorkspaceLeak(
        resources -> {
          float[] wrongDimension = new float[DIMENSIONS - 1];
          expectThrows(
              IllegalArgumentException.class,
              () -> Utils.createFloatMatrix(List.of(wrongDimension), DIMENSIONS, resources));
        });
  }

  @Test
  public void testFloatMatrixOutlivesItsBuilder() throws Throwable {
    assertNoWorkspaceLeak(
        resources -> {
          float[] expected = {1f, 2f, 3f, 4f};
          // createFloatMatrix has already closed its builder; the returned matrix must remain
          // usable.
          try (CuVSMatrix matrix =
              Utils.createFloatMatrix(List.of(expected), expected.length, resources)) {
            float[][] actual = new float[1][expected.length];
            matrix.toArray(actual);
            assertArrayEquals(expected, actual[0], 0f);
          }
        });
  }

  @Test
  public void testFailedCagraBuildReleasesDatasetOnBruteForceFallback() throws Throwable {
    // NN-descent rejects Linf before launching kernels. Final allocation balance must include
    // cleanup of both the rejected CAGRA dataset and the successful brute-force fallback.
    GPUSearchParams params =
        new GPUSearchParams.Builder()
            .withStrategy(GPUSearchParams.Strategy.CUSTOM)
            .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
            .withCuvsDistanceType(CuvsDistanceType.Linf)
            .build();
    float[][] vectors = deterministicVectors();
    ByteArrayOutputStream buildFailureLog = new ByteArrayOutputStream();
    assertNoWorkspaceLeak(
        resources -> {
          try (Directory directory = new ByteBuffersDirectory();
              PrintStream info = new PrintStream(buildFailureLog, true, StandardCharsets.UTF_8)) {
            IndexWriterConfig config =
                singleSegmentConfig(vectors.length, params).setInfoStream(info);
            try (IndexWriter writer = new IndexWriter(directory, config)) {
              addVectors(writer, vectors);
              writer.commit();
            }
            assertTrue(
                "the intended native build failure did not occur: " + buildFailureLog,
                buildFailureLog
                    .toString(StandardCharsets.UTF_8)
                    .contains("The metric for NN Descent"));
            // Closing the writer removes its thread-local wrapper, not the outer tracker.
            setCuVSResourcesInstance(new NonClosingResources(resources));
            assertPersistedBruteForceFallbackSearchesCorrectly(directory, vectors[QUERY_DOCUMENT]);
          }
        });
  }

  @Test
  public void testBruteForceSerializationFailureReleasesBuiltIndex() throws Throwable {
    BruteForceSerializationFailure outputFailure = new BruteForceSerializationFailure();
    GPUSearchParams bruteForceParams =
        new GPUSearchParams.Builder()
            .withIndexType(CuVS2510GPUVectorsWriter.IndexType.BRUTE_FORCE)
            .build();
    assertNoWorkspaceLeak(
        resources -> {
          try (MockDirectoryWrapper directory =
              new MockDirectoryWrapper(random(), new ByteBuffersDirectory())) {
            directory.failOn(outputFailure);
            try (IndexWriter writer =
                new IndexWriter(directory, singleSegmentConfig(DOCUMENT_COUNT, bruteForceParams))) {
              addVectors(writer, deterministicVectors());
              IOException thrown = expectThrows(IOException.class, writer::commit);
              assertSame(
                  "commit must preserve the output failure", outputFailure.exception, thrown);
              assertTrue("serialization failure was not exercised", outputFailure.triggered);
              writer.rollback();
            }
          }
        });
  }

  @Test
  public void testRepeatedCagraQueriesReleaseWorkspaceAllocations() throws Throwable {
    assumeTrue("cuVS not supported", isSupported());

    float[][] vectors = deterministicVectors();
    Path memoryCsv = createTempDir("cagra-query-memory").resolve("allocations.csv");

    try (Directory directory = new ByteBuffersDirectory()) {
      buildSingleSegmentIndex(directory, vectors);
      runRepeatedQueriesWithMemoryTracking(directory, vectors[QUERY_DOCUMENT], memoryCsv);
    }

    assertWorkspaceReleased(memoryCsv);
  }

  private static void buildSingleSegmentIndex(Directory directory, float[][] vectors)
      throws Exception {
    IndexWriterConfig config =
        singleSegmentConfig(vectors.length, new GPUSearchParams.Builder().build());
    try (IndexWriter writer = new IndexWriter(directory, config)) {
      addVectors(writer, vectors);
      writer.commit();
    }
  }

  private static IndexWriterConfig singleSegmentConfig(int documentCount, GPUSearchParams params) {
    return new IndexWriterConfig()
        .setCodec(CuVSCodecs.gpuSearch(params))
        .setMergePolicy(NoMergePolicy.INSTANCE)
        .setRAMBufferSizeMB(256)
        .setMaxBufferedDocs(documentCount + 1);
  }

  private static void addVectors(IndexWriter writer, float[][] vectors) throws IOException {
    for (float[] vector : vectors) {
      Document document = new Document();
      document.add(new KnnFloatVectorField(VECTOR_FIELD, vector, EUCLIDEAN));
      writer.addDocument(document);
    }
  }

  /** The index is genuinely built; only writing its bytes to Lucene is made to fail. */
  private static final class BruteForceSerializationFailure extends MockDirectoryWrapper.Failure {
    private final IOException exception = new IOException("injected index-output failure");
    private boolean triggered;

    @Override
    public void eval(MockDirectoryWrapper ignored) throws IOException {
      if (triggered == false && isSerializingBruteForceIndex()) {
        triggered = true;
        throw exception;
      }
    }

    private static boolean isSerializingBruteForceIndex() {
      return StackWalker.getInstance()
          .walk(
              frames ->
                  frames.anyMatch(
                      frame ->
                          frame.getClassName().equals(CuVS2510GPUVectorsWriter.class.getName())
                              && frame.getMethodName().equals("writeBruteForceIndex")));
    }
  }

  private static void assertPersistedBruteForceFallbackSearchesCorrectly(
      Directory directory, float[] queryVector) throws IOException {
    try (DirectoryReader reader = DirectoryReader.open(directory)) {
      LeafReader leaf = getOnlyLeafReader(reader);
      var vectorField = leaf.getFieldInfos().fieldInfo(VECTOR_FIELD);
      assertNotNull("segment is missing the vector field", vectorField);
      GPUIndex index = gpuReader(leaf).getCuvsIndexes().get(vectorField.number);
      assertNotNull("segment is missing its GPU index", index);
      assertNull("the rejected CAGRA build must not be persisted", index.getCagraIndex());
      assertNotNull("expected the brute-force fallback", index.getBruteforceIndex());

      TopKnnCollector collector = new TopKnnCollector(TOP_K, Integer.MAX_VALUE);
      leaf.searchNearestVectors(VECTOR_FIELD, queryVector, collector, null);
      assertSelfMatchAndUniqueResults(collector.topDocs().scoreDocs);
    }
  }

  @FunctionalInterface
  private interface TrackedOperation {
    void run(CuVSResources resources) throws Throwable;
  }

  private void assertNoWorkspaceLeak(TrackedOperation operation) throws Throwable {
    assumeTrue("cuVS not supported", isSupported());
    closeCuVSResourcesInstance();
    Path csv = createTempDir("matrix-lifecycle").resolve("allocations.csv");
    try (CuVSResources resources =
        CuVSResources.create(CuVSProvider.tempDirectory(), csv, Duration.ofMillis(1))) {
      setCuVSResourcesInstance(new NonClosingResources(resources));
      try {
        operation.run(resources);
      } finally {
        // Avoid creating a new resources handle after the writer/reader removes its wrapper.
        setCuVSResourcesInstance(null);
        closeCuVSResourcesInstance();
      }
    }
    assertWorkspaceReleased(csv);
  }

  private static void assertWorkspaceReleased(Path csv) throws Exception {
    MemoryTotals workspace = finalMemoryTotals(csv, "workspace");
    assertTrue("the tracker did not observe workspace allocations", workspace.allocatedBytes() > 0);
    assertEquals("operation retained workspace memory", 0, workspace.currentBytes());
    assertEquals(
        "workspace allocations and deallocations are unbalanced",
        workspace.allocatedBytes(),
        workspace.freedBytes());
  }

  private static void runRepeatedQueriesWithMemoryTracking(
      Directory directory, float[] queryVector, Path memoryCsv) throws Throwable {
    try (CuVSResources trackedResources =
        CuVSResources.create(CuVSProvider.tempDirectory(), memoryCsv, Duration.ofMillis(1))) {
      setCuVSResourcesInstance(new NonClosingResources(trackedResources));
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        LeafReader leaf = getOnlyLeafReader(reader);
        assertCagraIndexLoaded(leaf);
        for (int i = 0; i < QUERY_REPETITIONS; i++) {
          TopKnnCollector collector = new TopKnnCollector(TOP_K, Integer.MAX_VALUE);
          leaf.searchNearestVectors(VECTOR_FIELD, queryVector, collector, null);
          assertSelfMatchAndUniqueResults(collector.topDocs().scoreDocs);
        }
      } finally {
        // DirectoryReader closes the provider's wrapper. Remove any value left by an early failure
        // without closing the underlying tracker a second time.
        setCuVSResourcesInstance(null);
        closeCuVSResourcesInstance();
      }
    }
  }

  private static void assertCagraIndexLoaded(LeafReader leaf) throws IOException {
    assertNotNull("expected a CAGRA index", gpuReader(leaf).getCagraIndexForField(VECTOR_FIELD));
  }

  private static CuVS2510GPUVectorsReader gpuReader(LeafReader leaf) throws IOException {
    KnnVectorsReader vectorReader = ((CodecReader) leaf).getVectorReader();
    if (vectorReader instanceof PerFieldKnnVectorsFormat.FieldsReader fieldsReader) {
      vectorReader = fieldsReader.getFieldReader(VECTOR_FIELD);
    }
    assertTrue(
        "expected the cuVS GPU vectors reader", vectorReader instanceof CuVS2510GPUVectorsReader);
    return (CuVS2510GPUVectorsReader) vectorReader;
  }

  private static void assertSelfMatchAndUniqueResults(ScoreDoc[] hits) {
    assertEquals("unexpected result count", TOP_K, hits.length);
    assertEquals("the queried document must rank first", QUERY_DOCUMENT, hits[0].doc);
    HashSet<Integer> uniqueDocIds = new HashSet<>();
    for (ScoreDoc hit : hits) {
      assertTrue("duplicate result " + hit.doc, uniqueDocIds.add(hit.doc));
    }
  }

  private static float[][] deterministicVectors() {
    // This prime exceeds DOCUMENT_COUNT, giving each row a distinct first coordinate.
    // Normalizing the modular pattern keeps every component in [0, 1).
    long patternModulus = 1031L;
    float[][] vectors = new float[DOCUMENT_COUNT][DIMENSIONS];
    for (int row = 0; row < DOCUMENT_COUNT; row++) {
      for (int column = 0; column < DIMENSIONS; column++) {
        long rowMultiplier = row + 1L;
        long columnMultiplier = column + 3L;
        long patternValue = rowMultiplier * columnMultiplier % patternModulus;
        vectors[row][column] = patternValue / (float) patternModulus;
      }
    }
    return vectors;
  }

  private static MemoryTotals finalMemoryTotals(Path csv, String resourceName) throws Exception {
    List<String> lines = Files.readAllLines(csv);
    assertTrue("memory tracker did not write a sample", lines.size() >= 2);
    String[] header = lines.getFirst().split(",", -1);
    String[] finalSample = lines.getLast().split(",", -1);
    return new MemoryTotals(
        csvLong(header, finalSample, resourceName + "_current"),
        csvLong(header, finalSample, resourceName + "_total_alloc"),
        csvLong(header, finalSample, resourceName + "_total_freed"));
  }

  private static long csvLong(String[] header, String[] sample, String columnName) {
    for (int i = 0; i < header.length; i++) {
      if (columnName.equals(header[i])) {
        return Long.parseLong(sample[i]);
      }
    }
    fail("memory tracker CSV is missing column " + columnName);
    throw new AssertionError("unreachable");
  }

  private record MemoryTotals(long currentBytes, long allocatedBytes, long freedBytes) {}

  /**
   * Lets the reader close its indexes without stopping the memory tracker. The outer try-with-resources
   * closes the tracker last so its final CSV sample includes index cleanup.
   */
  private static final class NonClosingResources implements CuVSResources {
    private final CuVSResources delegate;

    private NonClosingResources(CuVSResources delegate) {
      this.delegate = delegate;
    }

    @Override
    public ScopedAccess access() {
      return delegate.access();
    }

    @Override
    public int deviceId() {
      return delegate.deviceId();
    }

    @Override
    public void close() {}

    @Override
    public void setWorkspacePool(long initialSizeBytes) {
      delegate.setWorkspacePool(initialSizeBytes);
    }

    @Override
    public Path tempDirectory() {
      return delegate.tempDirectory();
    }
  }
}

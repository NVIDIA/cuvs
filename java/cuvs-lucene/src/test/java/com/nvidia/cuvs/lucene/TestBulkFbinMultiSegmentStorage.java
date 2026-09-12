/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.CuVSTestSupport.enableRmmOrSkip;
import static com.nvidia.cuvs.lucene.CuVSTestSupport.requireCuvsOrSkip;
import static org.apache.lucene.index.VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;

import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressFileSystems;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.After;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

/** End-to-end coverage for merge-free mapped and external multi-segment bulk builds. */
@SuppressFileSystems("*")
@SuppressSysoutChecks(bugUrl = "")
public class TestBulkFbinMultiSegmentStorage extends LuceneTestCase {

  private static final String ID_FIELD = "id";
  private static final String CALLBACK_ID_FIELD = "callback_id";
  private static final String VECTOR_FIELD = "vector";
  private static final int ROWS = 302;
  private static final int DIMENSIONS = 32;
  private static final int SEGMENTS = 3;

  @BeforeClass
  public static void beforeClass() {
    enableRmmOrSkip();
  }

  @Before
  public void requireCuvs() {
    requireCuvsOrSkip();
  }

  @After
  public void clearRegistry() {
    ExternalFbinFileRegistry.clearForTests();
  }

  @Test
  public void testMappedBuildPreservesSlicesAndGlobalIds() throws Exception {
    Path root = createTempDir("mapped-multi-segment");
    float[][] vectors = createVectors();
    Path fbin = writeFbin(root, vectors);
    Path index = root.resolve("index");
    List<Integer> callbackIds = Collections.synchronizedList(new ArrayList<>());

    CagraHnswBulkIndexWriter.indexMappedFbin(
        fbin, config(index, new CagraHnswBuildMetrics()), callback(callbackIds));

    assertEquals(expectedIds(), callbackIds);
    assertEquals(SEGMENTS, countExtension(index, ".vec"));
    assertEquals(SEGMENTS, countExtension(index, ".vemf"));
    assertEquals(0, countExtension(index, ".vefr"));
    Files.delete(fbin);
    assertIndex(index, vectors);
  }

  @Test
  public void testMappedOverlappedBuildPublishesCompleteIndex() throws Exception {
    Path root = createTempDir("mapped-overlapped-multi-segment");
    float[][] vectors = createVectors();
    Path fbin = writeFbin(root, vectors);
    Path index = root.resolve("index");
    List<Integer> callbackIds = Collections.synchronizedList(new ArrayList<>());

    CagraHnswBulkIndexWriter.indexMappedFbin(
        fbin, config(index, new CagraHnswBuildMetrics(), true), callback(callbackIds));

    List<Integer> sortedCallbackIds = new ArrayList<>(callbackIds);
    Collections.sort(sortedCallbackIds);
    assertEquals(expectedIds(), sortedCallbackIds);
    Files.delete(fbin);
    assertIndex(index, vectors);
  }

  @Test
  public void testExternalBuildVerifiesOnceAndPreservesSlices() throws Exception {
    Path root = createTempDir("external-multi-segment");
    float[][] vectors = createVectors();
    Path fbin = writeFbin(root, vectors);
    Path index = root.resolve("index");
    String digest = sha256(fbin);
    CagraHnswBuildMetrics metrics = new CagraHnswBuildMetrics();
    List<Integer> callbackIds = Collections.synchronizedList(new ArrayList<>());

    try (ExternalFbinFileRegistry.Registration registration =
        ExternalFbinFileRegistry.register(fbin, digest)) {
      CagraHnswBulkIndexWriter.indexImmutableFbin(
          registration,
          config(index, metrics),
          ExternalFbinOptions.verifySha256(),
          callback(callbackIds));

      assertEquals(expectedIds(), callbackIds);
      assertEquals(SEGMENTS, countExtension(index, ".vefr"));
      assertEquals(0, countExtension(index, ".vec"));
      assertEquals(0, countExtension(index, ".vemf"));
      assertIndex(index, vectors);
    }

    Map<String, Number> snapshot = metrics.snapshot();
    assertEquals(1L, snapshot.get("stage/external fbin SHA-256 [DISK+CPU]/count").longValue());
    assertEquals(
        Files.size(fbin), snapshot.get("stage/external fbin SHA-256 [DISK+CPU]/bytes").longValue());
    assertEquals(SEGMENTS, snapshot.get("stage/base cagra-build [GPU]/count").longValue());
  }

  @Test
  public void testExternalIndexAndFbinCanRelocateAfterBuildLeaseCloses() throws Exception {
    Path root = createTempDir("external-relocation");
    float[][] vectors = createVectors();
    Path originalFbin = writeFbin(root, vectors);
    Path originalIndex = root.resolve("index-a");
    String digest = sha256(originalFbin);

    try (ExternalFbinFileRegistry.Registration registration =
        ExternalFbinFileRegistry.register(originalFbin, digest)) {
      CagraHnswBulkIndexWriter.indexImmutableFbin(
          registration,
          config(originalIndex, new CagraHnswBuildMetrics()),
          ExternalFbinOptions.verifySha256(),
          (document, id) -> document.add(new StoredField(CALLBACK_ID_FIELD, id)));
    }

    Path relocatedRoot = Files.createDirectory(root.resolve("relocated"));
    Path relocatedFbin = Files.move(originalFbin, relocatedRoot.resolve("vectors.fbin"));
    Path relocatedIndex = Files.move(originalIndex, relocatedRoot.resolve("index-b"));
    assertFalse(Files.exists(originalFbin));
    assertFalse(Files.exists(originalIndex));

    try (ExternalFbinFileRegistry.Registration relocatedLease =
        ExternalFbinFileRegistry.register(relocatedFbin, digest)) {
      assertIndex(relocatedIndex, vectors);
    }
  }

  @Test
  public void testExternalOverlappedBuildPrefetchesOnceAndPublishesCompleteIndex()
      throws Exception {
    Path root = createTempDir("external-overlapped-multi-segment");
    float[][] vectors = createVectors();
    Path fbin = writeFbin(root, vectors);
    Path index = root.resolve("index");
    String digest = sha256(fbin);
    CagraHnswBuildMetrics metrics = new CagraHnswBuildMetrics();
    List<Integer> callbackIds = Collections.synchronizedList(new ArrayList<>());

    try (ExternalFbinFileRegistry.Registration registration =
        ExternalFbinFileRegistry.register(fbin, digest)) {
      CagraHnswBulkIndexWriter.indexImmutableFbin(
          registration,
          config(index, metrics, true),
          new ExternalFbinOptions(ExternalFbinBuildValidation.PREFETCH, 0L),
          callback(callbackIds));

      List<Integer> sortedCallbackIds = new ArrayList<>(callbackIds);
      Collections.sort(sortedCallbackIds);
      assertEquals(expectedIds(), sortedCallbackIds);
      assertIndex(index, vectors);
    }

    Map<String, Number> snapshot = metrics.snapshot();
    assertEquals(1L, snapshot.get("stage/external fbin prefetch [DISK]/count").longValue());
    assertEquals(
        Files.size(fbin) - ExternalFbinReference.HEADER_BYTES,
        snapshot.get("stage/external fbin prefetch [DISK]/bytes").longValue());
    assertEquals(SEGMENTS, snapshot.get("stage/base cagra-build [GPU]/count").longValue());
  }

  @Test
  public void testExternalOverlappedBuildPreservesCallerInterruption() throws Exception {
    Path root = createTempDir("external-overlapped-interruption");
    Path fbin = writeFbin(root, createVectors());
    Path index = root.resolve("index");
    String digest = sha256(fbin);
    CountDownLatch callbackStarted = new CountDownLatch(1);
    CountDownLatch releaseCallback = new CountDownLatch(1);
    AtomicReference<Throwable> failure = new AtomicReference<>();
    AtomicBoolean interruptPreserved = new AtomicBoolean();

    try (ExternalFbinFileRegistry.Registration registration =
        ExternalFbinFileRegistry.register(fbin, digest)) {
      Thread buildThread =
          Thread.ofPlatform()
              .start(
                  () -> {
                    try {
                      CagraHnswBulkIndexWriter.indexImmutableFbin(
                          registration,
                          config(index, new CagraHnswBuildMetrics(), true),
                          new ExternalFbinOptions(ExternalFbinBuildValidation.PREFETCH, 0L),
                          (document, id) -> {
                            callbackStarted.countDown();
                            try {
                              releaseCallback.await();
                            } catch (InterruptedException interrupted) {
                              Thread.currentThread().interrupt();
                              throw new IOException("callback interrupted", interrupted);
                            }
                          });
                    } catch (Throwable thrown) {
                      failure.set(thrown);
                      interruptPreserved.set(Thread.currentThread().isInterrupted());
                    }
                  });

      assertTrue(callbackStarted.await(10, java.util.concurrent.TimeUnit.SECONDS));
      buildThread.interrupt();
      buildThread.join(10_000L);
      releaseCallback.countDown();
      assertFalse("interrupted build did not terminate", buildThread.isAlive());
    } finally {
      releaseCallback.countDown();
    }

    assertTrue(failure.get() instanceof java.io.InterruptedIOException);
    assertTrue(interruptPreserved.get());
    try (Directory directory = FSDirectory.open(index)) {
      assertFalse(DirectoryReader.indexExists(directory));
    }
  }

  @Test
  public void testExternalDigestFailureDoesNotPublishPartialIndex() throws Exception {
    Path root = createTempDir("external-multi-segment-failure");
    Path fbin = writeFbin(root, createVectors());
    Path index = root.resolve("index");
    String digest = sha256(fbin);
    corruptMiddleSlice(fbin);

    try (ExternalFbinFileRegistry.Registration registration =
        ExternalFbinFileRegistry.register(fbin, digest)) {
      IOException mismatch =
          expectThrows(
              IOException.class,
              () ->
                  CagraHnswBulkIndexWriter.indexImmutableFbin(
                      registration,
                      config(index, new CagraHnswBuildMetrics()),
                      ExternalFbinOptions.verifySha256()));
      assertTrue(mismatch.getMessage(), mismatch.getMessage().contains("SHA-256 mismatch"));
    }

    try (Directory directory = FSDirectory.open(index)) {
      assertFalse(DirectoryReader.indexExists(directory));
    }
    try (var children = Files.list(root)) {
      assertFalse(
          children.anyMatch(path -> path.getFileName().toString().startsWith("index.bulk-")));
    }
  }

  @Test
  public void testLaterSliceFailureDoesNotPublishPartialIndex() throws Exception {
    Path root = createTempDir("mapped-multi-segment-failure");
    Path fbin = writeFbin(root, createVectors());
    Path index = root.resolve("index");

    IOException failure =
        expectThrows(
            IOException.class,
            () ->
                CagraHnswBulkIndexWriter.indexMappedFbin(
                    fbin,
                    config(index, new CagraHnswBuildMetrics()),
                    (document, id) -> {
                      if (id == 150) {
                        throw new IOException("injected callback failure");
                      }
                    }));
    assertEquals("injected callback failure", failure.getMessage());

    try (Directory directory = FSDirectory.open(index)) {
      assertFalse(DirectoryReader.indexExists(directory));
    }
    try (var children = Files.list(root)) {
      assertFalse(
          children.anyMatch(path -> path.getFileName().toString().startsWith("index.bulk-")));
    }
  }

  private static CagraHnswBulkIndexWriter.FieldCallback callback(List<Integer> callbackIds) {
    return (document, id) -> {
      callbackIds.add(id);
      document.add(new StoredField(CALLBACK_ID_FIELD, id));
    };
  }

  private static CagraHnswBulkIndexWriter.Config config(Path index, CagraHnswBuildMetrics metrics) {
    return config(index, metrics, false);
  }

  private static CagraHnswBulkIndexWriter.Config config(
      Path index, CagraHnswBuildMetrics metrics, boolean overlapped) {
    AcceleratedHNSWParams graphBuild =
        new AcceleratedHNSWParams.Builder()
            .withCuvsDistanceType(CuvsDistanceType.InnerProduct)
            .build();
    return CagraHnswBulkIndexWriter.Config.builder()
        .field(VECTOR_FIELD, DIMENSIONS, MAXIMUM_INNER_PRODUCT)
        .idField(ID_FIELD)
        .graphBuild(graphBuild)
        .segments(SEGMENTS, overlapped)
        .targetDirectory(index)
        .metrics(metrics)
        .build();
  }

  private static void assertIndex(Path index, float[][] expectedVectors) throws Exception {
    int documents = 0;
    try (Directory directory = FSDirectory.open(index);
        DirectoryReader reader = DirectoryReader.open(directory)) {
      assertEquals(SEGMENTS, reader.leaves().size());
      for (var context : reader.leaves()) {
        LeafReader leaf = context.reader();
        FloatVectorValues values = leaf.getFloatVectorValues(VECTOR_FIELD);
        assertNotNull(values);
        assertEquals(leaf.maxDoc(), values.size());
        for (int localDoc = 0; localDoc < leaf.maxDoc(); localDoc++) {
          var stored = leaf.storedFields().document(localDoc);
          int globalId = Integer.parseInt(stored.get(ID_FIELD));
          assertEquals(globalId, context.docBase + localDoc);
          assertEquals(globalId, stored.getField(CALLBACK_ID_FIELD).numericValue().intValue());
          assertArrayEquals(expectedVectors[globalId], values.vectorValue(localDoc), 0.0f);
          documents++;
        }
        leaf.checkIntegrity();
      }
      assertEquals(ROWS, documents);

      float[] query = new float[DIMENSIONS];
      query[0] = 1.0f;
      IndexSearcher searcher = new IndexSearcher(reader);
      TopDocs hits = searcher.search(new KnnFloatVectorQuery(VECTOR_FIELD, query, 10), 10);
      assertEquals(10, hits.scoreDocs.length);
      assertEquals(
          Integer.toString(ROWS - 1),
          searcher.storedFields().document(hits.scoreDocs[0].doc).get(ID_FIELD));
    }
  }

  private static float[][] createVectors() {
    float[][] vectors = new float[ROWS][DIMENSIONS];
    Random random = new Random(8675309L);
    for (int row = 0; row < vectors.length; row++) {
      for (int dimension = 0; dimension < DIMENSIONS; dimension++) {
        vectors[row][dimension] = random.nextFloat() * 2.0f - 1.0f;
      }
    }
    vectors[ROWS - 1][0] = 100.0f;
    return vectors;
  }

  private static List<Integer> expectedIds() {
    List<Integer> ids = new ArrayList<>(ROWS);
    for (int id = 0; id < ROWS; id++) {
      ids.add(id);
    }
    return ids;
  }

  private static Path writeFbin(Path root, float[][] vectors) throws IOException {
    Path fbin = root.resolve("vectors.fbin");
    TestUtils.writeFbin(fbin, vectors);
    return fbin;
  }

  private static String sha256(Path file) throws Exception {
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    digest.update(Files.readAllBytes(file));
    return HexFormat.of().formatHex(digest.digest());
  }

  private static void corruptMiddleSlice(Path fbin) throws IOException {
    long rowBytes = (long) DIMENSIONS * Float.BYTES;
    long position = ExternalFbinReference.HEADER_BYTES + rowBytes * (ROWS / 2L);
    try (FileChannel channel = FileChannel.open(fbin, StandardOpenOption.WRITE)) {
      channel.write(ByteBuffer.wrap(new byte[] {42}), position);
    }
  }

  private static long countExtension(Path index, String extension) throws IOException {
    try (Directory directory = FSDirectory.open(index)) {
      return Arrays.stream(directory.listAll()).filter(name -> name.endsWith(extension)).count();
    }
  }
}

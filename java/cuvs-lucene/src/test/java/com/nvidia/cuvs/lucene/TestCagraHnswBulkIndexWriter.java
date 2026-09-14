/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.generateDataset;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import com.nvidia.cuvs.spi.CuVSProvider;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;
import java.util.UUID;
import org.apache.commons.io.FileUtils;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.After;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Functional coverage for {@link CagraHnswBulkIndexWriter}: the manual/direct-instance {@link
 * CagraHnswBulkIndexWriter#addDocument} API (including its safety checks) and the one-shot {@link
 * CagraHnswBulkIndexWriter#indexFbin}/{@link CagraHnswBulkIndexWriter#build(VectorSource, Config)}
 * convenience entry points (single-segment, K-segment sequential, K-segment overlapped,
 * {@link CagraHnswBulkIndexWriter.FieldCallback} metadata, rejection of {@code overlapped} for
 * {@code build}). Guard-rail behavior of the underlying native-buffered writer itself is already
 * covered by {@link TestNativeFlatBufferingGuardRails}; this class covers this class's own
 * orchestration and its own safety checks layered on top.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestCagraHnswBulkIndexWriter extends LuceneTestCase {

  private static final String ID_FIELD = "id";
  private static final String VECTOR_FIELD = "vector_field";
  private static final String CATEGORY_FIELD = "category";

  private Random random;
  private Path indexDirPath;
  private Path fbinPath;

  @BeforeClass
  public static void beforeClass() {
    // It is recommended to enable RMM allocation mode at application start, before constructing
    // any CagraHnswBulkIndexWriter, to avoid device-wide sync from the default allocator.
    try {
      CuVSProvider.provider().enableRMMAsyncMemory();
    } catch (UnsupportedOperationException unsupported) {
      assumeTrue("cuVS not supported: " + unsupported.getMessage(), false);
    }
  }

  @Before
  public void beforeTest() {
    assumeTrue("cuVS not supported", isSupported());
    random = new Random(222);
    indexDirPath = Paths.get(UUID.randomUUID().toString());
    fbinPath = Paths.get(UUID.randomUUID() + ".fbin");
  }

  @After
  public void afterTest() throws IOException {
    if (indexDirPath != null) {
      File dir = indexDirPath.toFile();
      if (dir.exists() && dir.isDirectory()) {
        FileUtils.deleteDirectory(dir);
      }
    }
    if (fbinPath != null) {
      new File(fbinPath.toString()).delete();
    }
  }

  @Test
  public void testSingleSegmentBuildIsSearchable() throws Exception {
    int numDocs = 300;
    int dimension = 32;
    float[][] dataset = generateDataset(random, numDocs, dimension);
    TestUtils.writeFbin(fbinPath, dataset);

    CagraHnswBulkIndexWriter.indexFbin(fbinPath, configFor(dimension, 1, false));

    assertSearchable(numDocs, dataset, /* expectedSegments= */ 1);
  }

  @Test
  public void testPartitionedSequentialBuildProducesKSegments() throws Exception {
    int numDocs = 400;
    int dimension = 24;
    int k = 4;
    float[][] dataset = generateDataset(random, numDocs, dimension);
    TestUtils.writeFbin(fbinPath, dataset);

    CagraHnswBulkIndexWriter.indexFbin(fbinPath, configFor(dimension, k, false));

    assertSearchable(numDocs, dataset, /* expectedSegments= */ k);
  }

  @Test
  public void testOverlappedBuildProducesKSegments() throws Exception {
    int numDocs = 400;
    int dimension = 24;
    int k = 4;
    float[][] dataset = generateDataset(random, numDocs, dimension);
    TestUtils.writeFbin(fbinPath, dataset);

    CagraHnswBulkIndexWriter.indexFbin(fbinPath, configFor(dimension, k, true));

    assertSearchable(numDocs, dataset, /* expectedSegments= */ k);
  }

  @Test
  public void testGenericBuildViaVectorSource() throws Exception {
    int numDocs = 200;
    int dimension = 16;
    float[][] dataset = generateDataset(random, numDocs, dimension);

    CagraHnswBulkIndexWriter.build(
        new InMemoryVectorSource(dataset), configFor(dimension, 1, false));

    assertSearchable(numDocs, dataset, /* expectedSegments= */ 1);
  }

  @Test
  public void testGenericBuildRejectsOverlapped() throws Exception {
    int dimension = 8;
    float[][] dataset = generateDataset(random, 50, dimension);
    CagraHnswBulkIndexWriter.Config config = configFor(dimension, 2, true);
    try {
      CagraHnswBulkIndexWriter.build(new InMemoryVectorSource(dataset), config);
      fail(
          "expected IllegalArgumentException: overlapped is not supported by build(VectorSource,"
              + " Config)");
    } catch (IllegalArgumentException expected) {
      // expected
    }
  }

  @Test
  public void testIdFieldCanBeDisabled() throws Exception {
    int numDocs = 50;
    int dimension = 8;
    float[][] dataset = generateDataset(random, numDocs, dimension);
    TestUtils.writeFbin(fbinPath, dataset);

    CagraHnswBulkIndexWriter.Config config =
        CagraHnswBulkIndexWriter.Config.builder()
            .field(VECTOR_FIELD, dimension, EUCLIDEAN)
            .idField(null)
            .graphBuild(new AcceleratedHNSWParams.Builder().build())
            .segments(1, false)
            .targetDirectory(indexDirPath)
            .build();
    CagraHnswBulkIndexWriter.indexFbin(fbinPath, config);

    try (Directory dir = FSDirectory.open(indexDirPath);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      TopDocs results = searcher.search(new KnnFloatVectorQuery(VECTOR_FIELD, dataset[0], 5), 5);
      assertEquals(5, results.scoreDocs.length);
      assertNull(
          "id field should not be stored when idField(null) is used",
          searcher.storedFields().document(results.scoreDocs[0].doc).get(ID_FIELD));
    }
  }

  /** {@link CagraHnswBulkIndexWriter.FieldCallback} lets the one-shot API attach metadata per row. */
  @Test
  public void testFieldCallbackAttachesMetadata() throws Exception {
    int numDocs = 60;
    int dimension = 8;
    float[][] dataset = generateDataset(random, numDocs, dimension);
    TestUtils.writeFbin(fbinPath, dataset);

    CagraHnswBulkIndexWriter.indexFbin(
        fbinPath,
        configFor(dimension, 1, false),
        (doc, id) ->
            doc.add(
                new StringField(CATEGORY_FIELD, id % 2 == 0 ? "even" : "odd", Field.Store.YES)));

    try (Directory dir = FSDirectory.open(indexDirPath);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      TopDocs results = searcher.search(new KnnFloatVectorQuery(VECTOR_FIELD, dataset[0], 1), 1);
      String category =
          searcher.storedFields().document(results.scoreDocs[0].doc).get(CATEGORY_FIELD);
      assertEquals("even", category); // id=0 is even
    }
  }

  /**
   * The manual/direct-instance API: caller builds the {@link Document} (including arbitrary extra
   * fields) and drives {@link CagraHnswBulkIndexWriter#addDocument} directly, same shape as a
   * plain {@link org.apache.lucene.index.IndexWriter}.
   */
  @Test
  public void testManualAddDocumentWithMetadata() throws Exception {
    int numDocs = 40;
    int dimension = 12;
    float[][] dataset = generateDataset(random, numDocs, dimension);

    CagraHnswBulkIndexWriter.Config config =
        CagraHnswBulkIndexWriter.Config.builder()
            .field(VECTOR_FIELD, dimension, EUCLIDEAN)
            .graphBuild(new AcceleratedHNSWParams.Builder().build())
            .build();

    try (Directory dir = FSDirectory.open(indexDirPath);
        CagraHnswBulkIndexWriter writer =
            new CagraHnswBulkIndexWriter(dir, new IndexWriterConfig(), config, numDocs)) {
      for (int i = 0; i < numDocs; i++) {
        Document doc = new Document();
        doc.add(new StringField(ID_FIELD, Integer.toString(i), Field.Store.YES));
        doc.add(new StringField(CATEGORY_FIELD, i % 2 == 0 ? "even" : "odd", Field.Store.YES));
        doc.add(new KnnFloatVectorField(VECTOR_FIELD, dataset[i], EUCLIDEAN));
        writer.addDocument(doc);
      }
    }

    try (Directory dir = FSDirectory.open(indexDirPath);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      assertEquals(1, reader.leaves().size());
      IndexSearcher searcher = new IndexSearcher(reader);
      TopDocs results = searcher.search(new KnnFloatVectorQuery(VECTOR_FIELD, dataset[0], 1), 1);
      assertEquals(
          "even", searcher.storedFields().document(results.scoreDocs[0].doc).get(CATEGORY_FIELD));
    }
  }

  @Test
  public void testCloseWithTooFewDocumentsThrows() throws Exception {
    int dimension = 8;
    CagraHnswBulkIndexWriter.Config config =
        CagraHnswBulkIndexWriter.Config.builder()
            .field(VECTOR_FIELD, dimension, EUCLIDEAN)
            .graphBuild(new AcceleratedHNSWParams.Builder().build())
            .build();

    try (Directory dir = FSDirectory.open(indexDirPath)) {
      CagraHnswBulkIndexWriter writer =
          new CagraHnswBulkIndexWriter(dir, new IndexWriterConfig(), config, 10);
      Document doc = new Document();
      doc.add(
          new KnnFloatVectorField(
              VECTOR_FIELD, generateDataset(random, 1, dimension)[0], EUCLIDEAN));
      writer.addDocument(doc); // only 1 of the promised 10

      try {
        writer.close();
        fail("expected IllegalStateException: fewer documents added than exactVectorCount");
      } catch (IllegalStateException expected) {
        // Specifically this class's mismatch error, not the underlying native writer's own count
        // check: close() must roll the buffered documents back, not flush them.
        assertEquals("expected 10 documents, got 1", expected.getMessage());
      }
      // The rollback left nothing committed, so there is no segment to open.
      assertFalse(DirectoryReader.indexExists(dir));
    }
  }

  @Test
  public void testSourceFailureMidBuildIsNotMaskedByCleanup() throws Exception {
    int dimension = 8;
    float[][] dataset = generateDataset(random, 200, dimension);
    // Fails halfway through, so the writer is cleaned up holding 100 of the 200 promised vectors.
    VectorSource failing = new FailingVectorSource(dataset, 100);

    try {
      CagraHnswBulkIndexWriter.build(failing, configFor(dimension, 1, false));
      fail("expected the source's IOException to propagate");
    } catch (IOException expected) {
      // The cleanup path must not report the count mismatch it necessarily sees instead of the
      // failure that caused it.
      assertEquals("source failed at 100", expected.getMessage());
    }
  }

  @Test
  public void testAddDocumentBeyondExactCountThrows() throws Exception {
    int dimension = 8;
    CagraHnswBulkIndexWriter.Config config =
        CagraHnswBulkIndexWriter.Config.builder()
            .field(VECTOR_FIELD, dimension, EUCLIDEAN)
            .graphBuild(new AcceleratedHNSWParams.Builder().build())
            .build();

    try (Directory dir = FSDirectory.open(indexDirPath)) {
      CagraHnswBulkIndexWriter writer =
          new CagraHnswBulkIndexWriter(dir, new IndexWriterConfig(), config, 1);
      float[][] vectors = generateDataset(random, 2, dimension);
      Document doc1 = new Document();
      doc1.add(new KnnFloatVectorField(VECTOR_FIELD, vectors[0], EUCLIDEAN));
      writer.addDocument(doc1);

      Document doc2 = new Document();
      doc2.add(new KnnFloatVectorField(VECTOR_FIELD, vectors[1], EUCLIDEAN));
      try {
        writer.addDocument(doc2); // exactVectorCount was 1
        fail("expected IllegalStateException: more documents added than exactVectorCount");
      } catch (IllegalStateException expected) {
        // expected -- the rejected call never reached the underlying writer or incremented the
        // count, so it's still exactly 1 and close() below completes normally.
      } finally {
        writer.close();
      }
    }
  }

  @Test
  public void testConstructorRejectsIndexSort() throws Exception {
    int dimension = 8;
    CagraHnswBulkIndexWriter.Config config =
        CagraHnswBulkIndexWriter.Config.builder()
            .field(VECTOR_FIELD, dimension, EUCLIDEAN)
            .graphBuild(new AcceleratedHNSWParams.Builder().build())
            .build();
    IndexWriterConfig conf =
        new IndexWriterConfig()
            .setIndexSort(new Sort(new SortField(ID_FIELD, SortField.Type.STRING)));

    try (Directory dir = FSDirectory.open(indexDirPath)) {
      try {
        new CagraHnswBulkIndexWriter(dir, conf, config, 10);
        fail("expected IllegalArgumentException: index-sorted segments are not supported");
      } catch (IllegalArgumentException expected) {
        // expected
      }
    }
  }

  private CagraHnswBulkIndexWriter.Config configFor(
      int dimension, int numSegments, boolean overlapped) {
    return CagraHnswBulkIndexWriter.Config.builder()
        .field(VECTOR_FIELD, dimension, EUCLIDEAN)
        .graphBuild(new AcceleratedHNSWParams.Builder().build())
        .segments(numSegments, overlapped)
        .targetDirectory(indexDirPath)
        .build();
  }

  private void assertSearchable(int numDocs, float[][] dataset, int expectedSegments)
      throws Exception {
    try (Directory dir = FSDirectory.open(indexDirPath);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      assertEquals("unexpected segment count", expectedSegments, reader.leaves().size());

      IndexSearcher searcher = new IndexSearcher(reader);
      int topK = 10;
      TopDocs results =
          searcher.search(new KnnFloatVectorQuery(VECTOR_FIELD, dataset[0], topK), topK);
      assertEquals(topK, results.scoreDocs.length);
      boolean sawQueryVectorItself = false;
      for (var scoreDoc : results.scoreDocs) {
        String id = searcher.storedFields().document(scoreDoc.doc).get(ID_FIELD);
        int idValue = Integer.parseInt(id);
        assertTrue("returned id out of range: " + id, idValue >= 0 && idValue < numDocs);
        sawQueryVectorItself |= idValue == 0;
      }
      // Querying with dataset[0] itself (an exact, zero-distance match) should reliably surface
      // id=0 within topK for a graph this small, across all segments the query fans out over.
      assertTrue(
          "expected id=0 (the query vector itself) within topK results", sawQueryVectorItself);
    }
  }

  /** An {@link InMemoryVectorSource} that throws once a given row is reached. */
  private static final class FailingVectorSource implements VectorSource {
    private final InMemoryVectorSource delegate;
    private final int failAt;

    FailingVectorSource(float[][] dataset, int failAt) {
      this.delegate = new InMemoryVectorSource(dataset);
      this.failAt = failAt;
    }

    @Override
    public int dimensions() {
      return delegate.dimensions();
    }

    @Override
    public int size() {
      return delegate.size();
    }

    @Override
    public void get(int index, float[] dst) throws IOException {
      if (index >= failAt) {
        throw new IOException("source failed at " + failAt);
      }
      delegate.get(index, dst);
    }

    @Override
    public void close() {
      delegate.close();
    }
  }

  private static final class InMemoryVectorSource implements VectorSource {
    private final float[][] dataset;

    InMemoryVectorSource(float[][] dataset) {
      this.dataset = dataset;
    }

    @Override
    public int dimensions() {
      return dataset.length == 0 ? 0 : dataset[0].length;
    }

    @Override
    public int size() {
      return dataset.length;
    }

    @Override
    public void get(int index, float[] dst) {
      System.arraycopy(dataset[index], 0, dst, 0, dst.length);
    }

    @Override
    public void close() {
      // nothing to release
    }
  }
}

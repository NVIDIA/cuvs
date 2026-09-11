/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;

import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import com.nvidia.cuvs.spi.CuVSProvider;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.Random;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
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

/** End-to-end storage and lifecycle coverage for the two borrowed-FBIN bulk build modes. */
@SuppressFileSystems("*")
@SuppressSysoutChecks(bugUrl = "")
public class TestBulkFbinIndexStorage extends LuceneTestCase {

  private static final String ID_FIELD = "id";
  private static final String VECTOR_FIELD = "vector";
  private static final int ROWS = 300;
  private static final int DIMENSIONS = 32;

  @BeforeClass
  public static void beforeClass() {
    try {
      CuVSProvider.provider().enableRMMAsyncMemory();
    } catch (UnsupportedOperationException unsupported) {
      assumeTrue("cuVS not supported: " + unsupported.getMessage(), false);
    }
  }

  @Before
  public void requireCuvs() {
    assumeTrue("cuVS not supported", isSupported());
  }

  @After
  public void clearRegistry() {
    ExternalFbinFileRegistry.clearForTests();
  }

  @Test
  public void testMappedFbinWritesSelfContainedMIPIndex() throws Exception {
    Path root = createTempDir("mapped-fbin-index");
    Path fbin = writeFbin(root, createMIPVectors());
    Path index = root.resolve("index");

    CagraHnswBulkIndexWriter.indexMappedFbin(fbin, config(index));

    assertContainsExtension(index, ".vec");
    assertContainsExtension(index, ".vemf");
    assertOmitsExtension(index, ".vefr");
    Files.delete(fbin);
    assertMIPIndexSearchable(index);
  }

  @Test
  public void testImmutableExternalFbinRequiresRegistrationAndDetectsMutation() throws Exception {
    Path root = createTempDir("external-fbin-index");
    float[][] vectors = createMIPVectors();
    Path fbin = writeFbin(root, vectors);
    Path index = root.resolve("index");
    String digest = sha256(fbin);

    try (ExternalFbinFileRegistry.Registration registration =
        ExternalFbinFileRegistry.register(fbin, digest)) {
      CagraHnswBulkIndexWriter.indexImmutableFbin(registration, config(index), trustedImmutable());

      assertContainsExtension(index, ".vefr");
      assertContainsExtension(index, ".vex");
      assertOmitsExtension(index, ".vec");
      assertOmitsExtension(index, ".vemf");
      assertMIPIndexSearchable(index);

      try (Directory directory = FSDirectory.open(index);
          DirectoryReader reader = DirectoryReader.open(directory)) {
        FloatVectorValues values = getOnlyLeafReader(reader).getFloatVectorValues(VECTOR_FIELD);
        assertArrayEquals(vectors[0], values.vectorValue(0), 0.0f);
        getOnlyLeafReader(reader).checkIntegrity();
      }
    }

    try (Directory directory = FSDirectory.open(index)) {
      IOException missingRegistration =
          expectThrows(IOException.class, () -> DirectoryReader.open(directory));
      assertTrue(
          missingRegistration.getMessage(),
          missingRegistration.getMessage().contains("No allowlisted external FBIN"));
    }

    corruptPayload(fbin);
    try (ExternalFbinFileRegistry.Registration ignored =
            ExternalFbinFileRegistry.register(fbin, digest);
        Directory directory = FSDirectory.open(index);
        DirectoryReader reader = DirectoryReader.open(directory)) {
      IOException corruptSource =
          expectThrows(IOException.class, () -> getOnlyLeafReader(reader).checkIntegrity());
      assertTrue(
          corruptSource.getMessage(), corruptSource.getMessage().contains("SHA-256 mismatch"));
    }
  }

  @Test
  public void testDigestMismatchDoesNotPublishCommit() throws Exception {
    Path root = createTempDir("external-fbin-validation");
    Path fbin = writeFbin(root, createMIPVectors());
    Path index = root.resolve("index");
    String digest = sha256(fbin);
    corruptPayload(fbin);

    try (ExternalFbinFileRegistry.Registration registration =
        ExternalFbinFileRegistry.register(fbin, digest)) {
      IOException mismatch =
          expectThrows(
              IOException.class,
              () ->
                  CagraHnswBulkIndexWriter.indexImmutableFbin(
                      registration, config(index), ExternalFbinOptions.verifySha256()));
      assertTrue(mismatch.getMessage(), mismatch.getMessage().contains("SHA-256 mismatch"));
    }

    try (Directory directory = FSDirectory.open(index)) {
      assertFalse(
          "A failed external-FBIN verification must not publish a commit",
          DirectoryReader.indexExists(directory));
    }
  }

  private static CagraHnswBulkIndexWriter.Config config(Path index) {
    AcceleratedHNSWParams graphBuild =
        new AcceleratedHNSWParams.Builder()
            .withCuvsDistanceType(CuvsDistanceType.InnerProduct)
            .build();
    return CagraHnswBulkIndexWriter.Config.builder()
        .field(VECTOR_FIELD, DIMENSIONS, MAXIMUM_INNER_PRODUCT)
        .idField(ID_FIELD)
        .graphBuild(graphBuild)
        .segments(1, false)
        .targetDirectory(index)
        .build();
  }

  private static ExternalFbinOptions trustedImmutable() {
    return new ExternalFbinOptions(ExternalFbinBuildValidation.TRUSTED_IMMUTABLE, 0L);
  }

  private static void assertMIPIndexSearchable(Path index) throws Exception {
    try (Directory directory = FSDirectory.open(index);
        DirectoryReader reader = DirectoryReader.open(directory)) {
      assertEquals(1, reader.leaves().size());
      float[] query = new float[DIMENSIONS];
      query[0] = 1.0f;
      IndexSearcher searcher = new IndexSearcher(reader);
      TopDocs hits = searcher.search(new KnnFloatVectorQuery(VECTOR_FIELD, query, 10), 10);

      assertEquals(10, hits.scoreDocs.length);
      assertEquals(0, hits.scoreDocs[0].doc);
      assertEquals(101.0f, hits.scoreDocs[0].score, 0.0f);
      assertEquals("0", searcher.storedFields().document(hits.scoreDocs[0].doc).get(ID_FIELD));
    }
  }

  private static float[][] createMIPVectors() {
    float[][] vectors = new float[ROWS][DIMENSIONS];
    vectors[0][0] = 100.0f;
    Random random = new Random(8675309L);
    for (int row = 1; row < vectors.length; row++) {
      for (int dimension = 0; dimension < DIMENSIONS; dimension++) {
        vectors[row][dimension] = random.nextFloat() * 2.0f - 1.0f;
      }
    }
    return vectors;
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

  private static void corruptPayload(Path fbin) throws IOException {
    try (FileChannel channel = FileChannel.open(fbin, StandardOpenOption.WRITE)) {
      channel.write(ByteBuffer.wrap(new byte[] {42}), ExternalFbinReference.HEADER_BYTES + 7L);
    }
  }

  private static void assertContainsExtension(Path index, String extension) throws IOException {
    try (Directory directory = FSDirectory.open(index)) {
      assertTrue(
          "Expected " + extension + " in " + index,
          Arrays.stream(directory.listAll()).anyMatch(name -> name.endsWith(extension)));
    }
  }

  private static void assertOmitsExtension(Path index, String extension) throws IOException {
    try (Directory directory = FSDirectory.open(index)) {
      assertFalse(
          "Did not expect " + extension + " in " + index,
          Arrays.stream(directory.listAll()).anyMatch(name -> name.endsWith(extension)));
    }
  }
}

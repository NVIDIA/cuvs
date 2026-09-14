/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.IOUtils;
import org.junit.After;

public class TestExternalFbinReference extends LuceneTestCase {

  @After
  public void clearRegistry() {
    ExternalFbinFileRegistry.clearForTests();
  }

  public void testRegisteredReferenceRoundTripAndVerification() throws Exception {
    Path file = writeFbin(createTempDir().resolve("vectors.fbin"), 4, 3);
    String sha256 = sha256(file);

    try (ExternalFbinFileRegistry.Registration registration =
        ExternalFbinFileRegistry.register(file, sha256)) {
      ExternalFbinReference reference = registration.reference(1, 2);

      assertEquals("sha256:" + sha256, reference.contentId());
      assertEquals(1L, reference.firstRow());
      assertEquals(2, reference.rows());
      assertEquals(3, reference.dimensions());
      assertEquals(8L + 3L * Float.BYTES, reference.payloadOffset());
      assertEquals(2L * 3 * Float.BYTES, reference.payloadLength());
      assertEquals(file.toRealPath(), ExternalFbinIO.validateAndResolve(reference));
      assertEquals(reference.fileLength(), ExternalFbinIO.verifySha256(reference));

      List<Long> prefetchProgress = new ArrayList<>();
      assertEquals(
          reference.payloadLength(), ExternalFbinIO.prefetch(reference, prefetchProgress::add));
      assertFalse(prefetchProgress.isEmpty());
      assertEquals(
          reference.payloadLength(), prefetchProgress.get(prefetchProgress.size() - 1).longValue());

      List<Long> verifyProgress = new ArrayList<>();
      assertEquals(
          reference.fileLength(), ExternalFbinIO.verifySha256(reference, verifyProgress::add));
      assertFalse(verifyProgress.isEmpty());
      assertEquals(
          reference.fileLength(), verifyProgress.get(verifyProgress.size() - 1).longValue());
    }
  }

  public void testScanHeadStartRequiresScanningValidationAndFitsPayload() throws Exception {
    Path root = Files.createTempDirectory("cuvs-external-fbin-params-");
    try {
      Path file = writeFbin(root.resolve("vectors.fbin"), 4, 3);
      String sha256 = sha256(file);

      try (ExternalFbinFileRegistry.Registration registration =
              ExternalFbinFileRegistry.register(file, sha256);
          ImmutableExternalFbinDataset dataset = registration.map(0, 4)) {
        IllegalArgumentException trustedFailure =
            expectThrows(
                IllegalArgumentException.class,
                () -> new ExternalFbinOptions(ExternalFbinBuildValidation.TRUSTED_IMMUTABLE, 1L));
        assertTrue(trustedFailure.getMessage().contains("PREFETCH or VERIFY_SHA256"));

        ExternalFbinOptions oversized =
            new ExternalFbinOptions(
                ExternalFbinBuildValidation.PREFETCH, dataset.reference().payloadLength() + 1L);
        IllegalArgumentException oversizedFailure =
            expectThrows(
                IllegalArgumentException.class,
                () ->
                    BulkIndexingContext.external(dataset, oversized, new CagraHnswBuildMetrics()));
        assertTrue(oversizedFailure.getMessage().contains("exceeds"));
      }
    } finally {
      IOUtils.rm(root);
    }
  }

  public void testVerifyHeadStartAccountsForNonzeroFirstRow() throws Exception {
    Path file = writeFbin(createTempDir().resolve("vectors.fbin"), 4, 3);
    try (ExternalFbinFileRegistry.Registration registration =
        ExternalFbinFileRegistry.register(file, sha256(file))) {
      ExternalFbinReference reference = registration.reference(1, 2);
      long requestedPayloadLead = (long) reference.dimensions() * Float.BYTES;
      CagraHnswBuildMetrics metrics = new CagraHnswBuildMetrics();
      ExternalFbinScanCoordinator coordinator =
          ExternalFbinScanCoordinator.start(
              reference, ExternalFbinBuildValidation.VERIFY_SHA256, requestedPayloadLead, metrics);

      coordinator.runAfterHeadStart(() -> {});

      Map<String, Number> snapshot = metrics.snapshot();
      assertEquals(
          requestedPayloadLead,
          snapshot.get("counter/external fbin requested head-start bytes").longValue());
      assertEquals(
          reference.payloadOffset() + requestedPayloadLead,
          snapshot.get("counter/external fbin required scan bytes").longValue());
    }
  }

  public void testMissingRegistrationFailsResolution() throws Exception {
    Path file = writeFbin(createTempDir().resolve("vectors.fbin"), 2, 2);
    String sha256 = sha256(file);
    ExternalFbinReference reference = ExternalFbinReference.fromFile(file, sha256, 0, 2);

    IOException failure =
        expectThrows(IOException.class, () -> ExternalFbinIO.validateAndResolve(reference));
    assertTrue(failure.getMessage().contains("No allowlisted external FBIN"));
  }

  public void testDigestMismatchFailsVerification() throws Exception {
    Path file = writeFbin(createTempDir().resolve("vectors.fbin"), 3, 2);
    String sha256 = sha256(file);

    try (ExternalFbinFileRegistry.Registration registration =
        ExternalFbinFileRegistry.register(file, sha256)) {
      ExternalFbinReference reference = registration.reference(0, 3);
      try (FileChannel channel =
          FileChannel.open(file, StandardOpenOption.WRITE, StandardOpenOption.READ)) {
        channel.write(ByteBuffer.wrap(new byte[] {99}), ExternalFbinReference.HEADER_BYTES + 1);
      }

      IOException failure =
          expectThrows(IOException.class, () -> ExternalFbinIO.verifySha256(reference));
      assertTrue(failure.getMessage().contains("SHA-256 mismatch"));
    }
  }

  public void testTruncatedFbinIsRejected() throws Exception {
    Path file = writeFbin(createTempDir().resolve("vectors.fbin"), 3, 2);
    String sha256 = sha256(file);
    try (FileChannel channel = FileChannel.open(file, StandardOpenOption.WRITE)) {
      channel.truncate(channel.size() - 1);
    }

    IOException failure =
        expectThrows(IOException.class, () -> ExternalFbinReference.fromFile(file, sha256, 0, 3));
    assertTrue(failure.getMessage().contains("does not match header-derived length"));
  }

  public void testAmbiguousContentRegistrationIsRejected() throws Exception {
    Path directory = createTempDir();
    Path first = writeFbin(directory.resolve("first.fbin"), 2, 2);
    Path second = Files.copy(first, directory.resolve("second.fbin"));
    String sha256 = sha256(first);

    try (ExternalFbinFileRegistry.Registration ignored =
        ExternalFbinFileRegistry.register(first, sha256)) {
      IllegalStateException failure =
          expectThrows(
              IllegalStateException.class, () -> ExternalFbinFileRegistry.register(second, sha256));
      assertTrue(failure.getMessage().contains("refusing ambiguous replacement"));
    }
  }

  public void testRegistrationLeasesAreReferenceCounted() throws Exception {
    Path file = writeFbin(createTempDir().resolve("vectors.fbin"), 2, 2);
    String sha256 = sha256(file);
    ExternalFbinFileRegistry.Registration first = ExternalFbinFileRegistry.register(file, sha256);
    ExternalFbinFileRegistry.Registration second = ExternalFbinFileRegistry.register(file, sha256);
    ExternalFbinReference reference = first.reference(0, 2);

    first.close();
    assertEquals(file.toRealPath(), ExternalFbinIO.validateAndResolve(reference));
    second.close();

    IOException failure =
        expectThrows(IOException.class, () -> ExternalFbinIO.validateAndResolve(reference));
    assertTrue(failure.getMessage().contains("No allowlisted external FBIN"));
  }

  private static Path writeFbin(Path file, int rows, int dimensions) throws IOException {
    ByteBuffer data =
        ByteBuffer.allocate(2 * Integer.BYTES + rows * dimensions * Float.BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
    data.putInt(rows);
    data.putInt(dimensions);
    for (int row = 0; row < rows; row++) {
      for (int dimension = 0; dimension < dimensions; dimension++) {
        data.putFloat(row * 10.0f + dimension);
      }
    }
    data.flip();
    try (FileChannel channel =
        FileChannel.open(file, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
      while (data.hasRemaining()) {
        channel.write(data);
      }
    }
    return file;
  }

  private static String sha256(Path file) throws Exception {
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    digest.update(Files.readAllBytes(file));
    return HexFormat.of().formatHex(digest.digest());
  }
}

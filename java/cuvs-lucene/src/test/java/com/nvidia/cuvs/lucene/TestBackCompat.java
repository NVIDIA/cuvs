/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import com.nvidia.cuvs.LibraryException;
import java.util.List;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.util.Version;
import org.junit.Test;

/**
 * Tests that the codecs from {@link CuVSCodecs} are registered, and that a cuvs-lucene artifact
 * on another Lucene release than it is built for fails only when its codecs are used.
 *
 * @since 25.12
 */
public class TestBackCompat {

  private static List<Codec> codecs() {
    return List.of(
        CuVSCodecs.acceleratedHNSW(),
        CuVSCodecs.acceleratedHNSWScalarQuantized(),
        CuVSCodecs.acceleratedHNSWBinaryQuantized(),
        CuVSCodecs.gpuSearch());
  }

  @Test
  public void testCodecsAreRegistered() {
    for (Codec codec : codecs()) {
      Codec registered = Codec.forName(codec.getName());
      assertSame(codec.getName(), codec.getClass(), registered.getClass());
    }
  }

  @Test
  public void testLuceneFormatsStillResolve() {
    // Listing a Lucene format in this module's services file used to break every lookup once a
    // later Lucene release moved that format.
    String name = "Lucene99HnswVectorsFormat";
    assertEquals(name, KnnVectorsFormat.forName(name).getName());
  }

  @Test
  public void testMismatchMessage() {
    assertNull(LuceneVersionGuard.mismatchMessage(Version.LATEST));
    String message = LuceneVersionGuard.mismatchMessage(Version.fromBits(9, 12, 1));
    assertNotNull(message);
    assertTrue(message, message.contains("Lucene 9.12.1 is in use"));
    assertTrue(message, message.contains("cuvs-lucene-9.12 artifact"));
    assertTrue(
        message,
        message.contains(
            "cuvs-lucene-" + LuceneCompat.LUCENE_MAJOR + "." + LuceneCompat.LUCENE_MINOR));
  }

  @Test
  public void testUnavailableDelegate() {
    // What a codec gets when its Lucene delegate does not link, e.g. on another Lucene release:
    // creating the codec must not throw, since Lucene's service loader does it for every codec.
    // The problem is logged once per JVM; mark it as logged, so that this test neither prints a
    // misleading SEVERE message nor uses up the one log a later test in this JVM might expect.
    boolean logged = CuVSFilterCodec.setUnavailableLoggedForTesting(true);
    try {
      Codec codec =
          CuVSFilterCodec.delegate(
              "TestCodec",
              () -> {
                throw new NoClassDefFoundError("org/apache/lucene/codecs/Missing");
              });
      assertEquals("TestCodec", codec.getName());
      IllegalStateException e = assertThrows(IllegalStateException.class, codec::postingsFormat);
      assertTrue(e.getMessage(), e.getMessage().contains("org/apache/lucene/codecs/Missing"));
      assertThrows(IllegalStateException.class, codec::knnVectorsFormat);
    } finally {
      CuVSFilterCodec.setUnavailableLoggedForTesting(logged);
    }
  }

  @Test
  public void testUnavailableFormat() {
    // What a codec gets when its vectors format does not link, e.g. because a later Lucene release
    // moved a class it uses, or cannot reach the cuVS library:
    // creating the codec must not throw either, for the same reason.
    checkUnavailableFormat(new NoClassDefFoundError("org/apache/lucene/codecs/Missing"));
    checkUnavailableFormat(new LibraryException("cuVS library not available"));
  }

  private static void checkUnavailableFormat(Throwable cause) {
    boolean logged = CuVSFilterCodec.setUnavailableLoggedForTesting(true);
    try {
      CuVSFilterCodec codec =
          new CuVSFilterCodec(
              "TestCodec",
              Codec.getDefault(),
              () -> {
                if (cause instanceof Error error) {
                  throw error;
                }
                throw (RuntimeException) cause;
              }) {};
      KnnVectorsFormat format = codec.knnVectorsFormat();
      IllegalStateException e =
          assertThrows(IllegalStateException.class, () -> format.fieldsWriter(null));
      assertTrue(e.getMessage(), e.getMessage().contains(cause.getMessage()));
      assertThrows(IllegalStateException.class, () -> format.fieldsReader(null));
      // Everything but vectors still works.
      assertNotNull(codec.postingsFormat());
    } finally {
      CuVSFilterCodec.setUnavailableLoggedForTesting(logged);
    }
  }

  @Test
  public void testLuceneVersionMatches() {
    // Loading the compatibility layer checks that this is the Lucene release it was built for.
    CuVSCodecs.checkLuceneVersion();
  }
}

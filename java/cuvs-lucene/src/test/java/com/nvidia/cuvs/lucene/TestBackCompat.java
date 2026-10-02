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

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.util.Version;
import org.junit.Test;

/**
 * Tests how the codecs of successive Lucene releases coexist: only the codecs from {@link
 * CuVSCodecs} write, and every older codec this Lucene release can read is registered, read-only.
 *
 * @since 25.12
 */
public class TestBackCompat {

  private static List<Codec> writableCodecs() {
    return List.of(
        CuVSCodecs.acceleratedHNSW(),
        CuVSCodecs.acceleratedHNSWScalarQuantized(),
        CuVSCodecs.acceleratedHNSWBinaryQuantized(),
        CuVSCodecs.gpuSearch());
  }

  private static List<CuVSFilterCodec> registeredCodecs() {
    List<CuVSFilterCodec> codecs = new ArrayList<>();
    for (String name : Codec.availableCodecs()) {
      if (Codec.forName(name) instanceof CuVSFilterCodec codec) {
        codecs.add(codec);
      }
    }
    return codecs;
  }

  @Test
  public void testWritableCodecsAreRegistered() {
    for (Codec codec : writableCodecs()) {
      Codec registered = Codec.forName(codec.getName());
      assertSame(codec.getName(), codec.getClass(), registered.getClass());
    }
  }

  @Test
  public void testOlderCodecsAreReadOnly() {
    Set<String> writable =
        writableCodecs().stream().map(Codec::getName).collect(Collectors.toSet());
    List<CuVSFilterCodec> registered = registeredCodecs();
    // One codec per family and Lucene codec generation this release can read.
    assertEquals(0, registered.size() % writable.size());
    for (CuVSFilterCodec codec : registered) {
      KnnVectorsFormat format = codec.knnVectorsFormat();
      assertNotNull(codec.getName(), format);
      if (writable.contains(codec.getName())) {
        continue;
      }
      UnsupportedOperationException e =
          assertThrows(UnsupportedOperationException.class, () -> format.fieldsWriter(null));
      assertTrue(e.getMessage(), e.getMessage().contains(codec.getName()));
    }
  }

  @Test
  public void testOldScalarQuantizedFormatNamesItsReplacement() {
    // Looked up by name, as a per-field configuration such as Solr's does. It stores its vectors
    // with Lucene99ScalarQuantizedVectorsFormat, which Lucene 10.4 moved to its backward codecs.
    BaseAcceleratedHNSWScalarQuantizedVectorsFormat format =
        (BaseAcceleratedHNSWScalarQuantizedVectorsFormat)
            KnnVectorsFormat.forName("Lucene99AcceleratedHNSWScalarQuantizedVectorsFormat");
    KnnVectorsFormat replacement =
        CuVSCodecs.acceleratedHNSWScalarQuantizedFormat(
            new AcceleratedHNSWParams.Builder().build());
    if (replacement.getName().equals(format.getName())) {
      // Still the format CuVSCodecs writes with, on Lucene 10.2 and 10.3.
      assertNull(format.readOnlyReason());
      return;
    }
    UnsupportedOperationException e =
        assertThrows(UnsupportedOperationException.class, () -> format.fieldsWriter(null));
    assertTrue(e.getMessage(), e.getMessage().contains(format.getName()));
    assertTrue(e.getMessage(), e.getMessage().contains(replacement.getName()));
  }

  @Test
  public void testCallerChosenDelegateIsNeverReadOnly() {
    // Lucene101Codec is a read-only backward codec from Lucene 10.3 on, but a codec built with a
    // delegate chosen by the caller leaves that to the caller instead of guessing from the class.
    // Created by class on purpose: CuVSCodecs cannot pass a delegate, and this constructor is what
    // is under test.
    Codec codec = new Lucene101AcceleratedHNSWCodec("CustomCodec", Codec.forName("Lucene101"));
    assertTrue(
        codec.knnVectorsFormat().toString(),
        codec.knnVectorsFormat() instanceof Lucene99AcceleratedHNSWVectorsFormat);
  }

  @Test
  public void testReadOnlyFormatIsNotWrappedTwice() {
    for (CuVSFilterCodec codec : registeredCodecs()) {
      KnnVectorsFormat format = codec.knnVectorsFormat();
      codec.setKnnFormat(format);
      // Handing back the format a read-only codec returned must not wrap it once more.
      assertSame(codec.getName(), format, codec.knnVectorsFormat());
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
  }

  @Test
  public void testLuceneVersionMatches() {
    // Loading the compatibility layer checks that this is the Lucene release it was built for.
    CuVSCodecs.checkLuceneVersion();
  }
}

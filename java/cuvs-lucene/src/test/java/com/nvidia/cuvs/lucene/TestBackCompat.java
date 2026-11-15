/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import com.nvidia.cuvs.LibraryException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.util.Version;
import org.junit.Test;

/**
 * Tests how the codecs of successive Lucene releases coexist: only the codecs from {@link
 * CuVSCodecs} write, and every older codec this Lucene release can read is registered, read-only.
 * Also tests that a cuvs-lucene artifact on another Lucene release than it is built for fails only
 * when its codecs are used.
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
      assertEquals(
          codec.getName(), writable.contains(codec.getName()) == false, codec.isReadOnly());
      if (writable.contains(codec.getName())) {
        continue;
      }
      UnsupportedOperationException e =
          assertThrows(UnsupportedOperationException.class, () -> format.fieldsWriter(null));
      assertTrue(e.getMessage(), e.getMessage().contains(codec.getName()));
    }
  }

  @Test
  public void testLucene99ScalarQuantizedFormatWritesOnlyWhereLuceneDoes() {
    // Created by class on purpose: CuVSCodecs returns it only on releases that can write it.
    BaseAcceleratedHNSWScalarQuantizedVectorsFormat format =
        new LuceneAcceleratedHNSWScalarQuantizedVectorsFormat();
    // A codec built with a writable delegate of the caller's own still writes only where the
    // format does.
    Codec custom = new LuceneAcceleratedHNSWScalarQuantizedCodec("CustomCodec", Codec.getDefault());
    KnnVectorsFormat customFormat = custom.knnVectorsFormat();
    assertFalse(customFormat.toString(), CuVSFilterCodec.isReadOnlyFormat(customFormat));
    // From the Lucene release, not from the rule the format applies, so that a module mapping
    // CuVSCodecs to the wrong format fails here: Lucene 10.4 moved its flat format to the backward
    // codecs.
    boolean writes = LuceneCompat.LUCENE_MAJOR == 10 && LuceneCompat.LUCENE_MINOR < 4;
    if (writes) {
      assertNull(format.readOnlyReason());
      assertNull(((BaseAcceleratedHNSWScalarQuantizedVectorsFormat) customFormat).readOnlyReason());
    } else {
      assertThrows(UnsupportedOperationException.class, () -> customFormat.fieldsWriter(null));
      // Thrown before the write state is used, naming the format to write with instead.
      UnsupportedOperationException e =
          assertThrows(UnsupportedOperationException.class, () -> format.fieldsWriter(null));
      assertTrue(e.getMessage(), e.getMessage().contains(format.getName()));
      assertTrue(
          e.getMessage(),
          e.getMessage().contains("CuVSCodecs.acceleratedHNSWScalarQuantizedFormat"));
      assertTrue(
          e.getMessage(),
          e.getMessage()
              .contains(BaseAcceleratedHNSWScalarQuantizedVectorsFormat.currentFormatName()));
    }
    assertEquals(
        writes ? format.getName() : "Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat",
        CuVSCodecs.acceleratedHNSWScalarQuantizedFormat().getName());
    assertEquals(
        BaseAcceleratedHNSWScalarQuantizedVectorsFormat.currentFormatName(),
        CuVSCodecs.acceleratedHNSWScalarQuantizedFormat().getName());
    // The scalar-quantized format CuVSCodecs returns always writes.
    assertNull(
        ((BaseAcceleratedHNSWScalarQuantizedVectorsFormat)
                CuVSCodecs.acceleratedHNSWScalarQuantizedFormat())
            .readOnlyReason());
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
      KnnVectorsFormat previous = codec.knnVectorsFormat();
      try {
        codec.setKnnVectorsFormat(previous);
        // Handing back the format a read-only codec returned must not wrap it once more.
        KnnVectorsFormat format = codec.knnVectorsFormat();
        assertEquals(
            format.toString(),
            CuVSFilterCodec.isReadOnlyFormat(previous),
            CuVSFilterCodec.isReadOnlyFormat(format));
        assertFalse(format.toString(), format.toString().startsWith("ReadOnly(ReadOnly("));
      } finally {
        // The registered codecs are shared by the whole JVM.
        codec.setKnnVectorsFormat(previous);
      }
    }
  }

  @Test
  public void testWritableCodecWritesWithFormatOfReadOnlyCodec() {
    for (CuVSFilterCodec old : registeredCodecs()) {
      if (old.isReadOnly() == false) {
        continue;
      }
      // Created through CuVSCodecs, not looked up, so that the registered codec stays unchanged.
      CuVSFilterCodec current = (CuVSFilterCodec) CuVSCodecs.acceleratedHNSW();
      assertFalse(current.getName(), current.isReadOnly());
      current.setKnnVectorsFormat(old.knnVectorsFormat());
      KnnVectorsFormat format = current.knnVectorsFormat();
      assertFalse(
          old.getName() + " made " + current.getName() + " read-only",
          CuVSFilterCodec.isReadOnlyFormat(format));
      // The codec no longer refuses to write, but a format Lucene can no longer write with still
      // does, on its own.
      if (format instanceof BaseAcceleratedHNSWScalarQuantizedVectorsFormat sq
          && sq.readOnlyReason() != null) {
        UnsupportedOperationException e =
            assertThrows(UnsupportedOperationException.class, () -> format.fieldsWriter(null));
        assertTrue(e.getMessage(), e.getMessage().contains(format.getName()));
      }
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
      Supplier<KnnVectorsFormat> format =
          () -> {
            if (cause instanceof Error error) {
              throw error;
            }
            throw (RuntimeException) cause;
          };
      checkUnavailableFormat(
          new CuVSFilterCodec("TestCodec", Codec.getDefault(), format) {}, cause);
      // A codec of an older generation, which can only read, still reports why its format is
      // unavailable rather than that it is read-only.
      checkUnavailableFormat(
          new CuVSFilterCodec("TestCodec", 0, () -> Codec.getDefault(), format) {}, cause);
    } finally {
      CuVSFilterCodec.setUnavailableLoggedForTesting(logged);
    }
  }

  private static void checkUnavailableFormat(CuVSFilterCodec codec, Throwable cause) {
    KnnVectorsFormat format = codec.knnVectorsFormat();
    IllegalStateException e =
        assertThrows(IllegalStateException.class, () -> format.fieldsWriter(null));
    assertTrue(e.getMessage(), e.getMessage().contains(cause.getMessage()));
    assertThrows(IllegalStateException.class, () -> format.fieldsReader(null));
    // Everything but vectors still works.
    assertNotNull(codec.postingsFormat());
  }

  @Test
  public void testLuceneVersionMatches() {
    // Loading the compatibility layer checks that this is the Lucene release it was built for.
    CuVSCodecs.checkLuceneVersion();
  }
}

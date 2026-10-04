/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.junit.Test;

/** Verifies that Lucene SPI discovery does not initialize optional vector-format dependencies. */
public class TestLuceneFormatLazyInitialization {

  private static final long PROBE_TIMEOUT_SECONDS = 30;
  private static final long TERMINATION_TIMEOUT_SECONDS = 5;

  @Test
  public void testCodecSpiDiscoveryDoesNotInitializeVectorFormatDependencies() throws Exception {
    assertFreshJvmProbeSucceeds(FreshJvmProbe.CODEC_SPI_MODE);
  }

  @Test
  public void testVectorFormatSpiDiscoveryDoesNotInitializeDependencies() throws Exception {
    assertFreshJvmProbeSucceeds(FreshJvmProbe.VECTOR_FORMAT_SPI_MODE);
  }

  @Test
  public void testVectorFormatConstructionDoesNotInitializeDependencies() throws Exception {
    assertFreshJvmProbeSucceeds(FreshJvmProbe.CONSTRUCTION_MODE);
  }

  @Test
  public void testLazyInitializationRetainsTheOriginalFailureCause() throws Exception {
    assertFreshJvmProbeSucceeds(FreshJvmProbe.FAILURE_CAUSE_MODE);
  }

  @Test
  public void testBinaryFormatRetainsReleasedLucene99FlatStorage() throws Exception {
    assertFreshJvmProbeSucceeds(FreshJvmProbe.BINARY_STORAGE_MODE);
  }

  @Test
  public void testConcurrentInitializationPublishesOneFlatFormat() throws Exception {
    assertFreshJvmProbeSucceeds(FreshJvmProbe.CONCURRENT_INITIALIZATION_MODE);
  }

  private static void assertFreshJvmProbeSucceeds(String mode) throws Exception {
    String classPath =
        System.getProperty("surefire.test.class.path", System.getProperty("java.class.path"));
    String javaExecutable =
        Path.of(System.getProperty("java.home"), "bin", "java").toAbsolutePath().toString();
    Path outputFile = Files.createTempFile("cuvs-lucene-lazy-init-" + mode + "-", ".log");
    Process process = null;
    try {
      process =
          new ProcessBuilder(
                  javaExecutable,
                  "--add-modules=jdk.incubator.vector",
                  "--enable-native-access=ALL-UNNAMED",
                  "-cp",
                  classPath,
                  FreshJvmProbe.class.getName(),
                  mode)
              .redirectErrorStream(true)
              .redirectOutput(outputFile.toFile())
              .start();

      boolean completed = process.waitFor(PROBE_TIMEOUT_SECONDS, TimeUnit.SECONDS);
      if (!completed) {
        process.destroyForcibly();
        boolean terminated = process.waitFor(TERMINATION_TIMEOUT_SECONDS, TimeUnit.SECONDS);
        throw new AssertionError(
            "Timed out waiting for "
                + mode
                + " fresh-JVM probe; terminated="
                + terminated
                + "; output:\n"
                + Files.readString(outputFile, StandardCharsets.UTF_8));
      }

      if (process.exitValue() != 0) {
        throw new AssertionError(
            mode
                + " fresh-JVM probe exited "
                + process.exitValue()
                + "; output:\n"
                + Files.readString(outputFile, StandardCharsets.UTF_8));
      }
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      throw interrupted;
    } finally {
      if (process != null && process.isAlive()) {
        process.destroyForcibly();
      }
      Files.deleteIfExists(outputFile);
    }
  }

  /** Runs assertions that require process-local class state untouched by other tests. */
  public static final class FreshJvmProbe {

    static final String CODEC_SPI_MODE = "codec-spi";
    static final String VECTOR_FORMAT_SPI_MODE = "vector-format-spi";
    static final String CONSTRUCTION_MODE = "construction";
    static final String FAILURE_CAUSE_MODE = "failure-cause";
    static final String BINARY_STORAGE_MODE = "binary-storage";
    static final String CONCURRENT_INITIALIZATION_MODE = "concurrent-initialization";

    private static final Set<String> CODEC_NAMES =
        Set.of(
            "Lucene101AcceleratedHNSWCodec",
            "CuVS2510GPUSearchCodec",
            "Lucene101AcceleratedHNSWBinaryQuantizedCodec",
            "Lucene101AcceleratedHNSWScalarQuantizedCodec");

    private static final Set<String> VECTOR_FORMAT_NAMES =
        Set.of(
            "CuVS2510GPUVectorsFormat",
            "Lucene99AcceleratedHNSWVectorsFormat",
            "Lucene99AcceleratedHNSWBinaryQuantizedVectorsFormat",
            "Lucene99AcceleratedHNSWScalarQuantizedVectorsFormat");

    private FreshJvmProbe() {}

    public static void main(String[] args) throws Exception {
      if (args.length != 1) {
        throw new IllegalArgumentException("Expected one probe mode");
      }
      switch (args[0]) {
        case CODEC_SPI_MODE -> assertCodecSpiDiscoveryIsLazy();
        case VECTOR_FORMAT_SPI_MODE -> assertVectorFormatSpiDiscoveryIsLazy();
        case CONSTRUCTION_MODE -> assertConstructionIsLazy();
        case FAILURE_CAUSE_MODE -> assertLazyFailureRetainsCause();
        case BINARY_STORAGE_MODE -> assertBinaryFormatUsesLucene99FlatStorage();
        case CONCURRENT_INITIALIZATION_MODE -> assertConcurrentInitializationSharesCache();
        default -> throw new IllegalArgumentException("Unknown probe mode: " + args[0]);
      }
    }

    private static void assertCodecSpiDiscoveryIsLazy() throws Exception {
      Set<String> available = Codec.availableCodecs();
      assertAllNamesAreAvailable("codecs", available, CODEC_NAMES);
      for (String name : CODEC_NAMES) {
        assertEquals(name, Codec.forName(name).getName());
      }
      assertLuceneProviderIsUninitialized();
    }

    private static void assertVectorFormatSpiDiscoveryIsLazy() throws Exception {
      Set<String> available = KnnVectorsFormat.availableKnnVectorsFormats();
      assertAllNamesAreAvailable("vector formats", available, VECTOR_FORMAT_NAMES);
      for (String name : VECTOR_FORMAT_NAMES) {
        assertEquals(name, KnnVectorsFormat.forName(name).getName());
      }
      assertLuceneProviderIsUninitialized();
    }

    private static void assertConstructionIsLazy() throws Exception {
      new Lucene99AcceleratedHNSWVectorsFormat();
      new CuVS2510GPUVectorsFormat();
      new LuceneAcceleratedHNSWScalarQuantizedVectorsFormat();
      new LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat();

      assertLuceneProviderIsUninitialized();
    }

    private static void assertLazyFailureRetainsCause() throws Exception {
      IOException originalFailure = new IOException("flat-format constructor failure");
      ThrowingFlatVectorsFormat.failure = originalFailure;

      LuceneProvider provider = LuceneProvider.getInstance("99");
      Field flatVectorsFormat = LuceneProvider.class.getDeclaredField("flatVectorsFormat");
      flatVectorsFormat.setAccessible(true);
      flatVectorsFormat.set(provider, ThrowingFlatVectorsFormat.class);

      KnnVectorsFormat format = new Lucene99AcceleratedHNSWVectorsFormat();
      Throwable thrown = captureFailure(() -> format.fieldsReader(null));

      if (thrown instanceof ExceptionInInitializerError) {
        throw new AssertionError("Lazy use must not fail through class initialization", thrown);
      }
      if (!containsCause(thrown, originalFailure)) {
        throw new AssertionError(
            "Original construction failure is absent from the cause chain", thrown);
      }
    }

    private static void assertBinaryFormatUsesLucene99FlatStorage() throws Exception {
      FlatVectorsFormat flatFormat =
          invokeFlatFormatFactory(LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.class);

      assertEquals(
          "org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat",
          flatFormat.getClass().getName());
    }

    private static void assertConcurrentInitializationSharesCache() throws Exception {
      int taskCount = 32;
      ExecutorService executor = Executors.newFixedThreadPool(8);
      List<Future<FlatVectorsFormat>> futures = new ArrayList<>(taskCount);
      try {
        for (int i = 0; i < taskCount; i++) {
          futures.add(
              executor.submit(
                  () -> invokeFlatFormatFactory(Lucene99AcceleratedHNSWVectorsFormat.class)));
        }

        Set<FlatVectorsFormat> identities =
            Collections.newSetFromMap(new IdentityHashMap<FlatVectorsFormat, Boolean>());
        for (Future<FlatVectorsFormat> future : futures) {
          identities.add(getFuture(future));
        }
        if (identities.size() != 1) {
          throw new AssertionError("Expected one cached flat format, found " + identities.size());
        }
      } finally {
        executor.shutdownNow();
        if (!executor.awaitTermination(5, TimeUnit.SECONDS)) {
          throw new AssertionError("Lazy-initialization workers did not terminate");
        }
      }
    }

    private static FlatVectorsFormat getFuture(Future<FlatVectorsFormat> future) throws Exception {
      try {
        return future.get();
      } catch (ExecutionException failure) {
        Throwable cause = failure.getCause();
        if (cause instanceof Exception exception) {
          throw exception;
        }
        if (cause instanceof Error error) {
          throw error;
        }
        throw new AssertionError("Unexpected worker failure", cause);
      }
    }

    private static FlatVectorsFormat invokeFlatFormatFactory(Class<?> formatClass)
        throws Exception {
      Method factory = formatClass.getDeclaredMethod("getOrCreateFlatVectorsFormat");
      factory.setAccessible(true);
      try {
        return (FlatVectorsFormat) factory.invoke(null);
      } catch (InvocationTargetException failure) {
        Throwable cause = failure.getCause();
        if (cause instanceof Exception exception) {
          throw exception;
        }
        if (cause instanceof Error error) {
          throw error;
        }
        throw new AssertionError("Unexpected lazy-initialization failure", cause);
      }
    }

    private static void assertLuceneProviderIsUninitialized() throws Exception {
      Field instance = LuceneProvider.class.getDeclaredField("instance");
      instance.setAccessible(true);
      if (instance.get(null) != null) {
        throw new AssertionError("Vector-format construction initialized LuceneProvider");
      }
    }

    private static void assertAllNamesAreAvailable(
        String kind, Set<String> available, Set<String> expected) {
      if (!available.containsAll(expected)) {
        Set<String> missing = new java.util.HashSet<>(expected);
        missing.removeAll(available);
        throw new AssertionError("Missing " + kind + ": " + missing + "; available=" + available);
      }
    }

    private static void assertEquals(String expected, String actual) {
      if (!expected.equals(actual)) {
        throw new AssertionError("Expected " + expected + " but found " + actual);
      }
    }

    private static Throwable captureFailure(ThrowingOperation operation) {
      try {
        operation.run();
      } catch (Throwable failure) {
        return failure;
      }
      throw new AssertionError("Expected operation to fail");
    }

    private static boolean containsCause(Throwable thrown, Throwable expected) {
      for (Throwable cause = thrown; cause != null; cause = cause.getCause()) {
        if (cause == expected) {
          return true;
        }
      }
      return false;
    }
  }

  @FunctionalInterface
  private interface ThrowingOperation {
    void run() throws Exception;
  }

  /** Supplies a deterministic reflective-construction failure to the lazy factory. */
  public static final class ThrowingFlatVectorsFormat extends FlatVectorsFormat {
    private static IOException failure;

    public ThrowingFlatVectorsFormat(FlatVectorsScorer ignored) throws IOException {
      super("ThrowingFlatVectorsFormat");
      throw failure;
    }

    @Override
    public FlatVectorsWriter fieldsWriter(SegmentWriteState state) {
      throw new AssertionError("Constructor failure should prevent writer creation");
    }

    @Override
    public FlatVectorsReader fieldsReader(SegmentReadState state) {
      throw new AssertionError("Constructor failure should prevent reader creation");
    }
  }
}

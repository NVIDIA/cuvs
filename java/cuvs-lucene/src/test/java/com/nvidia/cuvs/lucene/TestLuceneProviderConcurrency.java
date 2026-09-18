/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.junit.Assert.assertEquals;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.junit.Test;

/** Verifies that concurrent cold-start requests cannot mix version-specific providers. */
public class TestLuceneProviderConcurrency {

  private static final String PROBE_ARGUMENT = "--provider-concurrency-probe";
  private static final long PROBE_TIMEOUT_SECONDS = 30;
  private static final long TERMINATION_TIMEOUT_SECONDS = 5;

  @Test
  public void testConcurrentColdStartPublishesOneProviderPerVersion() throws Exception {
    String javaExecutable =
        Path.of(System.getProperty("java.home"), "bin", "java").toAbsolutePath().toString();
    String testClassPath =
        System.getProperty("surefire.test.class.path", System.getProperty("java.class.path"));
    Path outputFile = Files.createTempFile("cuvs-lucene-provider-concurrency-", ".log");
    Process process = null;
    try {
      process =
          new ProcessBuilder(
                  javaExecutable,
                  "--add-modules=jdk.incubator.vector",
                  "--enable-native-access=ALL-UNNAMED",
                  "-cp",
                  testClassPath,
                  TestLuceneProviderConcurrency.class.getName(),
                  PROBE_ARGUMENT)
              .redirectErrorStream(true)
              .redirectOutput(outputFile.toFile())
              .start();

      boolean completed = process.waitFor(PROBE_TIMEOUT_SECONDS, TimeUnit.SECONDS);
      if (!completed) {
        process.destroyForcibly();
        boolean terminated = process.waitFor(TERMINATION_TIMEOUT_SECONDS, TimeUnit.SECONDS);
        throw new AssertionError(
            "Timed out waiting for LuceneProvider concurrency probe; terminated="
                + terminated
                + "; output:\n"
                + Files.readString(outputFile, StandardCharsets.UTF_8));
      }

      String output = Files.readString(outputFile, StandardCharsets.UTF_8);
      assertEquals("Concurrency probe output:\n" + output, 0, process.exitValue());
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

  public static void main(String[] args) throws Exception {
    if (args.length != 1 || !PROBE_ARGUMENT.equals(args[0])) {
      throw new IllegalArgumentException("Expected " + PROBE_ARGUMENT);
    }
    runConcurrentColdStartProbe();
  }

  private static void runConcurrentColdStartProbe() throws Exception {
    int requestCount = 128;
    CountDownLatch start = new CountDownLatch(1);
    ExecutorService executor = Executors.newFixedThreadPool(16);
    Throwable primaryFailure = null;
    try {
      List<Future<ProviderResult>> futures = new ArrayList<>(requestCount);
      for (int request = 0; request < requestCount; request++) {
        String version =
            request % 2 == 0
                ? LuceneProvider.LUCENE_99_FORMAT_VERSION
                : LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION;
        futures.add(
            executor.submit(
                () -> {
                  start.await();
                  return new ProviderResult(version, LuceneProvider.getInstance(version));
                }));
      }

      start.countDown();
      LuceneProvider lucene99 = null;
      LuceneProvider lucene102 = null;
      for (Future<ProviderResult> future : futures) {
        ProviderResult result = future.get(20, TimeUnit.SECONDS);
        if (LuceneProvider.LUCENE_99_FORMAT_VERSION.equals(result.version())) {
          lucene99 = requireSameProvider(lucene99, result.provider(), result.version());
        } else {
          lucene102 = requireSameProvider(lucene102, result.provider(), result.version());
        }
      }

      if (lucene99 == null || lucene102 == null || lucene99 == lucene102) {
        throw new AssertionError("Expected distinct Lucene99 and Lucene102 providers");
      }
      lucene99.getLuceneFlatVectorsFormatInstance(DefaultFlatVectorScorer.INSTANCE);
      lucene102.getLuceneBinaryQuantizedVectorsFormatInstance();
    } catch (Exception | Error failure) {
      primaryFailure = failure;
      throw failure;
    } finally {
      executor.shutdownNow();
      try {
        if (!executor.awaitTermination(TERMINATION_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
          AssertionError terminationFailure =
              new AssertionError("LuceneProvider concurrency workers did not terminate");
          if (primaryFailure == null) {
            throw terminationFailure;
          }
          primaryFailure.addSuppressed(terminationFailure);
        }
      } catch (InterruptedException interrupted) {
        Thread.currentThread().interrupt();
        if (primaryFailure == null) {
          throw interrupted;
        }
        primaryFailure.addSuppressed(interrupted);
      }
    }
  }

  private static LuceneProvider requireSameProvider(
      LuceneProvider expected, LuceneProvider actual, String version) {
    if (expected != null && expected != actual) {
      throw new AssertionError("Observed multiple providers for Lucene version " + version);
    }
    return actual;
  }

  private record ProviderResult(String version, LuceneProvider provider) {}
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.junit.Test;

/**
 * Tests the backward compatibility mechanism.
 *
 * @since 25.12
 */
public class TestBackCompat {

  @Test
  public void testFallback() throws Exception {
    // Lucene99Codec exists in the org.apache.lucene.backward_codecs.lucene99
    Codec c = LuceneProvider.getCodec("99");
    assertEquals(c.getName(), "Lucene99");
  }

  @Test(expected = ClassNotFoundException.class)
  public void testNonexistentCodec() throws Exception {
    LuceneProvider.getCodec("0");
  }

  @Test
  public void testExistingComponents() throws Exception {
    LuceneProvider provider = LuceneProvider.getInstance(LuceneProvider.LUCENE_99_FORMAT_VERSION);
    assertTrue(provider.getLuceneFlatVectorsFormatInstance(null) instanceof FlatVectorsFormat);
    assertEquals(provider.getStaticIntParam("VERSION_CURRENT"), 0);
    assertNotEquals(provider.getSimilarityFunctions().size(), 0);
  }

  @Test
  public void testProviderCachesSupportedVersion() throws Exception {
    LuceneProvider lucene99Provider =
        LuceneProvider.getInstance(LuceneProvider.LUCENE_99_FORMAT_VERSION);
    assertSame(
        lucene99Provider, LuceneProvider.getInstance(LuceneProvider.LUCENE_99_FORMAT_VERSION));
  }

  @Test
  @SuppressWarnings("deprecation")
  public void testProviderSupportsLucene102BinaryFormats() throws Exception {
    LuceneProvider lucene102BinaryFormatProvider =
        LuceneProvider.getInstance(LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION);
    assertNotNull(lucene102BinaryFormatProvider.getLuceneBinaryQuantizedVectorsFormatInstance());
    assertNotNull(lucene102BinaryFormatProvider.getluceneBinaryQuantizedVectorsFormatInstance());
    assertNotNull(
        lucene102BinaryFormatProvider.getLuceneHnswBinaryQuantizedKnnVectorsFormatInstance(
            16, 100));
  }

  @Test
  public void testProviderSupportsLucene99ScalarFormats() throws Exception {
    LuceneProvider lucene99Provider =
        LuceneProvider.getInstance(LuceneProvider.LUCENE_99_FORMAT_VERSION);
    assertNotNull(lucene99Provider.getLuceneScalarQuantizedVectorsFormatInstance());
    KnnVectorsFormat hnswScalarFormat =
        lucene99Provider.getLuceneHnswScalarQuantizedKnnVectorsFormatInstance(16, 100);
    assertTrue(hnswScalarFormat.toString().contains("maxConn=16, beamWidth=100"));
  }

  @Test(expected = UnsupportedOperationException.class)
  @SuppressWarnings("deprecation")
  public void testLegacyHnswBinaryFormatDescriptorIsRetained() throws Exception {
    LuceneProvider.getInstance(LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION)
        .getLuceneHnswBinaryQuantizedVectorsFormatInstance(16, 100);
  }

  @Test(expected = UnsupportedOperationException.class)
  @SuppressWarnings("deprecation")
  public void testLegacyHnswScalarFormatDescriptorIsRetained() throws Exception {
    LuceneProvider.getInstance(LuceneProvider.LUCENE_99_FORMAT_VERSION)
        .getLuceneHnswScalarQuantizedVectorsFormatInstance(100, 16);
  }

  @Test
  public void testLucene101DelegateCodec() throws Exception {
    Codec delegate = LuceneProvider.getCodec("101");
    assertEquals("Lucene101", delegate.getName());
    assertEquals(
        "org.apache.lucene.codecs.lucene101.Lucene101Codec", delegate.getClass().getName());
  }

  @Test
  public void testServiceLoadedCodecsCanBeInstantiated() {
    String[] codecNames = {
      "Lucene101AcceleratedHNSWCodec",
      "CuVS2510GPUSearchCodec",
      "Lucene101AcceleratedHNSWBinaryQuantizedCodec",
      "Lucene101AcceleratedHNSWScalarQuantizedCodec"
    };
    for (String codecName : codecNames) {
      assertTrue(Codec.availableCodecs().contains(codecName));
      assertEquals(codecName, Codec.forName(codecName).getName());
    }
  }

  @Test
  public void testCodecSPIColdStart() throws Exception {
    runColdStartProbe("codec");
  }

  @Test
  public void testKnnVectorsFormatSPIColdStart() throws Exception {
    runColdStartProbe("knn");
  }

  @Test
  public void testScalarFormatConstructionIsLazy() throws Exception {
    runColdStartProbe("scalar-constructor");
  }

  private static void runColdStartProbe(String mode) throws Exception {
    String javaExecutable =
        Path.of(System.getProperty("java.home"), "bin", "java").toAbsolutePath().toString();
    String testClassPath =
        System.getProperty("surefire.test.class.path", System.getProperty("java.class.path"));
    Path outputFile = Files.createTempFile("cuvs-lucene-spi-" + mode + "-", ".log");
    try {
      Process process =
          new ProcessBuilder(
                  javaExecutable,
                  "--add-modules=jdk.incubator.vector",
                  "--enable-native-access=ALL-UNNAMED",
                  "-cp",
                  testClassPath,
                  SPIColdStartProbe.class.getName(),
                  mode)
              .redirectErrorStream(true)
              .redirectOutput(outputFile.toFile())
              .start();

      boolean completed = process.waitFor(30, TimeUnit.SECONDS);
      if (!completed) {
        process.destroyForcibly();
        process.waitFor(5, TimeUnit.SECONDS);
        throw new AssertionError("Timed out waiting for " + mode + " SPI cold-start probe");
      }

      String output = Files.readString(outputFile, StandardCharsets.UTF_8);
      assertEquals("Cold-start probe output:\n" + output, 0, process.exitValue());
    } finally {
      Files.deleteIfExists(outputFile);
    }
  }
}

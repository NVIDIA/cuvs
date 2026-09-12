/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.junit.Test;
import org.junit.function.ThrowingRunnable;

/**
 * Verifies Lucene format-version adaptation and retained legacy entry points.
 *
 * @since 25.12
 */
public class TestBackCompat {

  @Test
  public void testLucene99CodecLoadsFromBackwardCodecs() throws Exception {
    Codec codec = LuceneProvider.getCodec(LuceneProvider.LUCENE_99_FORMAT_VERSION);

    assertEquals("Lucene99", codec.getName());
    assertEquals(
        "org.apache.lucene.backward_codecs.lucene99.Lucene99Codec", codec.getClass().getName());
  }

  @Test(expected = ClassNotFoundException.class)
  public void testMissingCodecVersionThrowsClassNotFoundException() throws Exception {
    LuceneProvider.getCodec("0");
  }

  @Test
  public void testLucene99ProviderLoadsRequiredHnswComponents() throws Exception {
    LuceneProvider provider = LuceneProvider.getInstance(LuceneProvider.LUCENE_99_FORMAT_VERSION);

    assertTrue(
        provider.getLuceneFlatVectorsFormatInstance(DefaultFlatVectorScorer.INSTANCE)
            instanceof FlatVectorsFormat);
    assertEquals(0, provider.getStaticIntParam("VERSION_CURRENT"));
    assertFalse(provider.getSimilarityFunctions().isEmpty());
  }

  @Test
  public void testProviderCachesEachFormatVersionSeparately() throws Exception {
    LuceneProvider lucene99Provider =
        LuceneProvider.getInstance(LuceneProvider.LUCENE_99_FORMAT_VERSION);
    LuceneProvider lucene102Provider =
        LuceneProvider.getInstance(LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION);

    assertSame(
        lucene99Provider, LuceneProvider.getInstance(LuceneProvider.LUCENE_99_FORMAT_VERSION));
    assertSame(
        lucene102Provider,
        LuceneProvider.getInstance(LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION));
    assertNotSame(lucene99Provider, lucene102Provider);
  }

  @Test
  public void testLucene102ProviderConstructsFlatBinaryQuantizedFormat() throws Exception {
    LuceneProvider provider =
        LuceneProvider.getInstance(LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION);

    assertNotNull(provider.getLuceneBinaryQuantizedVectorsFormatInstance());
  }

  @Test
  @SuppressWarnings("deprecation")
  public void testLegacyMisspelledBinaryFormatFactoryRetainsDescriptorAndDelegates()
      throws Exception {
    LuceneProvider provider =
        LuceneProvider.getInstance(LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION);
    FlatVectorsFormat canonicalFormat = provider.getLuceneBinaryQuantizedVectorsFormatInstance();

    assertEquals(
        FlatVectorsFormat.class,
        LuceneProvider.class
            .getMethod("getluceneBinaryQuantizedVectorsFormatInstance")
            .getReturnType());
    assertEquals(
        canonicalFormat.getClass(),
        provider.getluceneBinaryQuantizedVectorsFormatInstance().getClass());
  }

  @Test
  public void testLucene102ProviderPreservesBinaryHnswParameterOrder() throws Exception {
    LuceneProvider provider =
        LuceneProvider.getInstance(LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION);

    KnnVectorsFormat format =
        provider.getLuceneHnswBinaryQuantizedKnnVectorsFormatInstance(16, 100);

    assertTrue(
        "Unexpected binary HNSW parameters: " + format,
        format.toString().contains("maxConn=16, beamWidth=100"));
  }

  @Test
  public void testLucene99ProviderConstructsFlatScalarQuantizedFormat() throws Exception {
    LuceneProvider provider = LuceneProvider.getInstance(LuceneProvider.LUCENE_99_FORMAT_VERSION);

    assertNotNull(provider.getLuceneScalarQuantizedVectorsFormatInstance());
  }

  @Test
  public void testLucene99ProviderPreservesScalarHnswParameterOrder() throws Exception {
    LuceneProvider provider = LuceneProvider.getInstance(LuceneProvider.LUCENE_99_FORMAT_VERSION);

    KnnVectorsFormat format =
        provider.getLuceneHnswScalarQuantizedKnnVectorsFormatInstance(16, 100);

    assertTrue(
        "Unexpected scalar HNSW parameters: " + format,
        format.toString().contains("maxConn=16, beamWidth=100"));
  }

  @Test
  @SuppressWarnings("deprecation")
  public void testLucene99ProviderRejectsBinaryQuantizedCapabilities() throws Exception {
    LuceneProvider provider = LuceneProvider.getInstance(LuceneProvider.LUCENE_99_FORMAT_VERSION);

    assertCapabilityIsUnavailable(
        "Lucene binary-quantized vectors",
        LuceneProvider.LUCENE_99_FORMAT_VERSION,
        LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION,
        provider::getLuceneBinaryQuantizedVectorsFormatInstance);
    assertCapabilityIsUnavailable(
        "Lucene binary-quantized vectors",
        LuceneProvider.LUCENE_99_FORMAT_VERSION,
        LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION,
        provider::getluceneBinaryQuantizedVectorsFormatInstance);
    assertCapabilityIsUnavailable(
        "Lucene HNSW binary-quantized vectors",
        LuceneProvider.LUCENE_99_FORMAT_VERSION,
        LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION,
        () -> provider.getLuceneHnswBinaryQuantizedKnnVectorsFormatInstance(16, 100));
  }

  @Test
  public void testLucene102ProviderRejectsGeneralAndScalarCapabilities() throws Exception {
    LuceneProvider provider =
        LuceneProvider.getInstance(LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION);

    assertCapabilityIsUnavailable(
        "Lucene flat vectors",
        LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION,
        LuceneProvider.LUCENE_99_FORMAT_VERSION,
        () -> provider.getLuceneFlatVectorsFormatInstance(null));
    assertCapabilityIsUnavailable(
        "Lucene HNSW vectors reader",
        LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION,
        LuceneProvider.LUCENE_99_FORMAT_VERSION,
        () -> provider.getLuceneHnswVectorsReaderInstance(null, null));
    assertCapabilityIsUnavailable(
        "Lucene HNSW vectors writer",
        LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION,
        LuceneProvider.LUCENE_99_FORMAT_VERSION,
        () -> provider.getLuceneHnswVectorsWriterInstance(null, 16, 100, null, 1, null));
    assertCapabilityIsUnavailable(
        "Lucene HNSW vectors",
        LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION,
        LuceneProvider.LUCENE_99_FORMAT_VERSION,
        () -> provider.getStaticIntParam("VERSION_CURRENT"));
    assertCapabilityIsUnavailable(
        "Lucene HNSW vectors reader",
        LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION,
        LuceneProvider.LUCENE_99_FORMAT_VERSION,
        provider::getSimilarityFunctions);
    assertCapabilityIsUnavailable(
        "Lucene scalar-quantized vectors",
        LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION,
        LuceneProvider.LUCENE_99_FORMAT_VERSION,
        provider::getLuceneScalarQuantizedVectorsFormatInstance);
    assertCapabilityIsUnavailable(
        "Lucene HNSW scalar-quantized vectors",
        LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION,
        LuceneProvider.LUCENE_99_FORMAT_VERSION,
        () -> provider.getLuceneHnswScalarQuantizedKnnVectorsFormatInstance(16, 100));
  }

  @Test
  @SuppressWarnings("deprecation")
  public void testLegacyHnswBinaryDescriptorIsRetainedWithActionableFailure() throws Exception {
    assertEquals(
        FlatVectorsFormat.class,
        LuceneProvider.class
            .getMethod("getLuceneHnswBinaryQuantizedVectorsFormatInstance", int.class, int.class)
            .getReturnType());

    UnsupportedOperationException thrown =
        assertThrows(
            UnsupportedOperationException.class,
            () ->
                LuceneProvider.getInstance(LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION)
                    .getLuceneHnswBinaryQuantizedVectorsFormatInstance(16, 100));

    assertTrue(
        thrown.getMessage().contains("getLuceneHnswBinaryQuantizedKnnVectorsFormatInstance"));
  }

  @Test
  @SuppressWarnings("deprecation")
  public void testLegacyHnswScalarDescriptorIsRetainedWithActionableFailure() throws Exception {
    assertEquals(
        FlatVectorsFormat.class,
        LuceneProvider.class
            .getMethod("getLuceneHnswScalarQuantizedVectorsFormatInstance", int.class, int.class)
            .getReturnType());

    UnsupportedOperationException thrown =
        assertThrows(
            UnsupportedOperationException.class,
            () ->
                LuceneProvider.getInstance(LuceneProvider.LUCENE_99_FORMAT_VERSION)
                    .getLuceneHnswScalarQuantizedVectorsFormatInstance(100, 16));

    assertTrue(
        thrown.getMessage().contains("getLuceneHnswScalarQuantizedKnnVectorsFormatInstance"));
  }

  @Test
  public void testLucene101CodecResolvesAsCurrentDelegate() throws Exception {
    Codec delegate = LuceneProvider.getCodec("101");

    assertEquals("Lucene101", delegate.getName());
    assertEquals(
        "org.apache.lucene.codecs.lucene101.Lucene101Codec", delegate.getClass().getName());
  }

  private static void assertCapabilityIsUnavailable(
      String capability,
      String providerVersion,
      String requiredVersion,
      ThrowingRunnable operation) {
    UnsupportedOperationException thrown =
        assertThrows(UnsupportedOperationException.class, operation);

    assertTrue(thrown.getMessage().contains(capability));
    assertTrue(thrown.getMessage().contains("version " + providerVersion));
    assertTrue(
        thrown.getMessage().contains("LuceneProvider.getInstance(\"" + requiredVersion + "\")"));
  }
}

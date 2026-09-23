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

import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
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

  private static final int NON_DEFAULT_MAX_CONN = 17;
  private static final int NON_DEFAULT_BEAM_WIDTH = 101;

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
  public void testProviderCachesEachFormatVersionSeparately() throws Exception {
    LuceneProvider lucene99 = LuceneProvider.getInstance(LuceneProvider.LUCENE_99_FORMAT_VERSION);
    LuceneProvider lucene102 =
        LuceneProvider.getInstance(LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION);

    assertSame(lucene99, LuceneProvider.getInstance(LuceneProvider.LUCENE_99_FORMAT_VERSION));
    assertSame(
        lucene102, LuceneProvider.getInstance(LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION));
    assertNotSame(lucene99, lucene102);
  }

  @Test
  public void testLucene99ProviderConstructsGeneralAndScalarFormats() throws Exception {
    LuceneProvider provider = LuceneProvider.getInstance(LuceneProvider.LUCENE_99_FORMAT_VERSION);

    assertEquals(
        "org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat",
        provider
            .getLuceneFlatVectorsFormatInstance(DefaultFlatVectorScorer.INSTANCE)
            .getClass()
            .getName());
    assertEquals(
        "org.apache.lucene.codecs.lucene99.Lucene99ScalarQuantizedVectorsFormat",
        provider.getLuceneScalarQuantizedVectorsFormatInstance().getClass().getName());
    assertEquals(0, provider.getStaticIntParam("VERSION_CURRENT"));
    assertFalse(provider.getSimilarityFunctions().isEmpty());
  }

  @Test
  public void testLucene102ProviderConstructsBinaryFormats() throws Exception {
    LuceneProvider provider =
        LuceneProvider.getInstance(LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION);

    assertEquals(
        "org.apache.lucene.codecs.lucene102.Lucene102BinaryQuantizedVectorsFormat",
        provider.getLuceneBinaryQuantizedVectorsFormatInstance().getClass().getName());
    assertEquals(
        "org.apache.lucene.codecs.lucene102.Lucene102HnswBinaryQuantizedVectorsFormat",
        provider
            .getLuceneHnswBinaryQuantizedKnnVectorsFormatInstance(16, 100)
            .getClass()
            .getName());
  }

  @Test
  public void testHnswQuantizedFactoriesPreserveMaxConnAndBeamWidthOrder() throws Exception {
    KnnVectorsFormat binary =
        LuceneProvider.getInstance(LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION)
            .getLuceneHnswBinaryQuantizedKnnVectorsFormatInstance(
                NON_DEFAULT_MAX_CONN, NON_DEFAULT_BEAM_WIDTH);
    KnnVectorsFormat scalar =
        LuceneProvider.getInstance(LuceneProvider.LUCENE_99_FORMAT_VERSION)
            .getLuceneHnswScalarQuantizedKnnVectorsFormatInstance(
                NON_DEFAULT_MAX_CONN, NON_DEFAULT_BEAM_WIDTH);

    assertTrue("Unexpected binary HNSW parameters: " + binary, hasExpectedParameters(binary));
    assertTrue("Unexpected scalar HNSW parameters: " + scalar, hasExpectedParameters(scalar));
  }

  @Test
  @SuppressWarnings("deprecation")
  public void testMisspelledBinaryFactoryRemainsADelegatingAlias() throws Exception {
    LuceneProvider provider =
        LuceneProvider.getInstance(LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION);

    assertEquals(
        provider.getLuceneBinaryQuantizedVectorsFormatInstance().getClass(),
        provider.getluceneBinaryQuantizedVectorsFormatInstance().getClass());
  }

  @Test
  @SuppressWarnings("deprecation")
  public void testLegacyHnswFactoryDescriptorsRemainLinkableAndFailActionably() throws Throwable {
    MethodHandles.Lookup lookup = MethodHandles.publicLookup();
    assertNotNull(
        lookup.findVirtual(
            LuceneProvider.class,
            "getLuceneHnswBinaryQuantizedVectorsFormatInstance",
            MethodType.methodType(FlatVectorsFormat.class, int.class, int.class)));
    assertNotNull(
        lookup.findVirtual(
            LuceneProvider.class,
            "getLuceneHnswScalarQuantizedVectorsFormatInstance",
            MethodType.methodType(FlatVectorsFormat.class, int.class, int.class)));

    UnsupportedOperationException binaryFailure =
        assertThrows(
            UnsupportedOperationException.class,
            () ->
                LuceneProvider.getInstance(LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION)
                    .getLuceneHnswBinaryQuantizedVectorsFormatInstance(16, 100));
    UnsupportedOperationException scalarFailure =
        assertThrows(
            UnsupportedOperationException.class,
            () ->
                LuceneProvider.getInstance(LuceneProvider.LUCENE_99_FORMAT_VERSION)
                    .getLuceneHnswScalarQuantizedVectorsFormatInstance(100, 16));

    assertTrue(
        binaryFailure
            .getMessage()
            .contains("getLuceneHnswBinaryQuantizedKnnVectorsFormatInstance"));
    assertTrue(
        scalarFailure
            .getMessage()
            .contains("getLuceneHnswScalarQuantizedKnnVectorsFormatInstance"));
  }

  @Test
  public void testProvidersRejectCapabilitiesOwnedByTheOtherFormatVersion() throws Exception {
    LuceneProvider lucene99 = LuceneProvider.getInstance(LuceneProvider.LUCENE_99_FORMAT_VERSION);
    LuceneProvider lucene102 =
        LuceneProvider.getInstance(LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION);

    assertCapabilityIsUnavailable(
        "Lucene binary-quantized vectors",
        LuceneProvider.LUCENE_99_FORMAT_VERSION,
        LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION,
        lucene99::getLuceneBinaryQuantizedVectorsFormatInstance);
    assertCapabilityIsUnavailable(
        "Lucene flat vectors",
        LuceneProvider.LUCENE_102_BINARY_FORMAT_VERSION,
        LuceneProvider.LUCENE_99_FORMAT_VERSION,
        () -> lucene102.getLuceneFlatVectorsFormatInstance(DefaultFlatVectorScorer.INSTANCE));
  }

  private static boolean hasExpectedParameters(KnnVectorsFormat format) {
    return format
        .toString()
        .contains("maxConn=" + NON_DEFAULT_MAX_CONN + ", beamWidth=" + NON_DEFAULT_BEAM_WIDTH);
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

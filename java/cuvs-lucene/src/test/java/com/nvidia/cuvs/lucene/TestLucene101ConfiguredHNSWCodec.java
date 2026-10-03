/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.Lucene101ConfiguredHNSWCodec.BEAM_WIDTH_PROPERTY;
import static com.nvidia.cuvs.lucene.Lucene101ConfiguredHNSWCodec.MAX_CONN_PROPERTY;

import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/** Tests the no-argument, property-configured accelerated HNSW codec contract. */
public class TestLucene101ConfiguredHNSWCodec extends LuceneTestCase {

  private static final String ACCELERATED_CODEC_NAME = "Lucene101AcceleratedHNSWCodec";
  private static final String ACCELERATED_FORMAT_NAME = "Lucene99AcceleratedHNSWVectorsFormat";

  private final List<Lucene101ConfiguredHNSWCodec> configuredCodecs = new ArrayList<>();
  private String previousMaxConn;
  private String previousBeamWidth;

  @Before
  public void saveAndClearConfigurationProperties() {
    previousMaxConn = System.clearProperty(MAX_CONN_PROPERTY);
    previousBeamWidth = System.clearProperty(BEAM_WIDTH_PROPERTY);
  }

  @After
  public void restoreConfigurationProperties() {
    for (Lucene101ConfiguredHNSWCodec codec : configuredCodecs) {
      codec.configuredParameters().getMergeExec().shutdownNow();
    }
    restoreProperty(MAX_CONN_PROPERTY, previousMaxConn);
    restoreProperty(BEAM_WIDTH_PROPERTY, previousBeamWidth);
  }

  @Test
  public void testPublicNoArgumentConstructorSnapshotsValidProperties() throws Exception {
    setConfiguration(16, 64);

    Lucene101ConfiguredHNSWCodec codec = newConfiguredCodec();

    assertTrue(Modifier.isFinal(codec.getClass().getModifiers()));
    assertTrue(Modifier.isPublic(codec.getClass().getConstructor().getModifiers()));
    assertEquals(16, codec.configuredParameters().getMaxConn());
    assertEquals(64, codec.configuredParameters().getBeamWidth());
  }

  @Test
  public void testSupportedRangeBoundariesAreAccepted() throws Exception {
    setConfiguration(AcceleratedHNSWParams.MIN_MAX_CONN, AcceleratedHNSWParams.MIN_BEAM_WIDTH);
    Lucene101ConfiguredHNSWCodec minimumCodec = newConfiguredCodec();

    setConfiguration(AcceleratedHNSWParams.MAX_MAX_CONN, AcceleratedHNSWParams.MAX_BEAM_WIDTH);
    Lucene101ConfiguredHNSWCodec maximumCodec = newConfiguredCodec();

    assertEquals(
        AcceleratedHNSWParams.MIN_MAX_CONN, minimumCodec.configuredParameters().getMaxConn());
    assertEquals(
        AcceleratedHNSWParams.MIN_BEAM_WIDTH, minimumCodec.configuredParameters().getBeamWidth());
    assertEquals(
        AcceleratedHNSWParams.MAX_MAX_CONN, maximumCodec.configuredParameters().getMaxConn());
    assertEquals(
        AcceleratedHNSWParams.MAX_BEAM_WIDTH, maximumCodec.configuredParameters().getBeamWidth());
  }

  @Test
  public void testEachMissingPropertyNamesTheRequiredConfiguration() {
    assertMissingProperty(MAX_CONN_PROPERTY);

    System.setProperty(MAX_CONN_PROPERTY, "16");
    assertMissingProperty(BEAM_WIDTH_PROPERTY);
  }

  @Test
  public void testEachMalformedPropertyReportsItsNameAndValue() {
    assertMalformedProperty(MAX_CONN_PROPERTY, "sixteen");
    assertMalformedProperty(BEAM_WIDTH_PROPERTY, "sixty-four");
  }

  @Test
  public void testEachPropertyRejectsValuesOutsideItsSupportedRange() {
    assertOutOfRangeProperty(
        MAX_CONN_PROPERTY,
        AcceleratedHNSWParams.MIN_MAX_CONN - 1,
        AcceleratedHNSWParams.MIN_MAX_CONN,
        AcceleratedHNSWParams.MAX_MAX_CONN);
    assertOutOfRangeProperty(
        MAX_CONN_PROPERTY,
        AcceleratedHNSWParams.MAX_MAX_CONN + 1,
        AcceleratedHNSWParams.MIN_MAX_CONN,
        AcceleratedHNSWParams.MAX_MAX_CONN);
    assertOutOfRangeProperty(
        BEAM_WIDTH_PROPERTY,
        AcceleratedHNSWParams.MIN_BEAM_WIDTH - 1,
        AcceleratedHNSWParams.MIN_BEAM_WIDTH,
        AcceleratedHNSWParams.MAX_BEAM_WIDTH);
    assertOutOfRangeProperty(
        BEAM_WIDTH_PROPERTY,
        AcceleratedHNSWParams.MAX_BEAM_WIDTH + 1,
        AcceleratedHNSWParams.MIN_BEAM_WIDTH,
        AcceleratedHNSWParams.MAX_BEAM_WIDTH);
  }

  @Test
  public void testEachInstanceRetainsItsOwnPropertySnapshot() throws Exception {
    setConfiguration(16, 48);
    Lucene101ConfiguredHNSWCodec first = newConfiguredCodec();

    setConfiguration(24, 96);
    Lucene101ConfiguredHNSWCodec second = newConfiguredCodec();

    assertEquals(16, first.configuredParameters().getMaxConn());
    assertEquals(48, first.configuredParameters().getBeamWidth());
    assertEquals(24, second.configuredParameters().getMaxConn());
    assertEquals(96, second.configuredParameters().getBeamWidth());
  }

  @Test
  public void testConfiguredCodecPreservesPersistedCodecAndVectorFormatNames() throws Exception {
    setConfiguration(16, 64);

    Lucene101ConfiguredHNSWCodec codec = newConfiguredCodec();

    assertEquals(ACCELERATED_CODEC_NAME, codec.getName());
    assertNotNull(
        "configured codec did not initialize its vector format", codec.knnVectorsFormat());
    assertEquals(ACCELERATED_FORMAT_NAME, codec.knnVectorsFormat().getName());
  }

  @Test
  public void testBaseCodecIgnoresConfiguredCodecProperties() throws Exception {
    System.setProperty(MAX_CONN_PROPERTY, "not-an-integer");
    System.setProperty(BEAM_WIDTH_PROPERTY, "also-not-an-integer");

    Lucene101AcceleratedHNSWCodec codec = new Lucene101AcceleratedHNSWCodec();

    assertEquals(Lucene101AcceleratedHNSWCodec.class, codec.getClass());
    assertEquals(ACCELERATED_CODEC_NAME, codec.getName());
    assertNotNull("base codec did not initialize its vector format", codec.knnVectorsFormat());
  }

  @Test
  public void testConfiguredCodecDoesNotReplaceLuceneSpiProvider() {
    assertEquals(
        Lucene101AcceleratedHNSWCodec.class, Codec.forName(ACCELERATED_CODEC_NAME).getClass());
  }

  private Lucene101ConfiguredHNSWCodec newConfiguredCodec() throws Exception {
    Lucene101ConfiguredHNSWCodec codec = new Lucene101ConfiguredHNSWCodec();
    configuredCodecs.add(codec);
    return codec;
  }

  private static void assertMissingProperty(String missingProperty) {
    IllegalStateException error =
        expectThrows(IllegalStateException.class, Lucene101ConfiguredHNSWCodec::new);
    assertTrue(error.getMessage().contains(missingProperty));
  }

  private static void assertMalformedProperty(String propertyName, String malformedValue) {
    setValidConfigurationForOtherProperty(propertyName);
    System.setProperty(propertyName, malformedValue);

    IllegalArgumentException error =
        expectThrows(IllegalArgumentException.class, Lucene101ConfiguredHNSWCodec::new);

    assertTrue(error.getMessage().contains(propertyName));
    assertTrue(error.getMessage().contains(malformedValue));
    assertNotNull("malformed integer error should retain its cause", error.getCause());
  }

  private static void assertOutOfRangeProperty(
      String propertyName, int value, int minimum, int maximum) {
    setValidConfigurationForOtherProperty(propertyName);
    System.setProperty(propertyName, Integer.toString(value));

    IllegalArgumentException error =
        expectThrows(IllegalArgumentException.class, Lucene101ConfiguredHNSWCodec::new);

    assertTrue(error.getMessage().contains(propertyName));
    assertTrue(error.getMessage().contains(Integer.toString(value)));
    assertTrue(error.getMessage().contains("[" + minimum + ", " + maximum + "]"));
  }

  private static void setValidConfigurationForOtherProperty(String propertyName) {
    if (MAX_CONN_PROPERTY.equals(propertyName)) {
      System.setProperty(BEAM_WIDTH_PROPERTY, "64");
    } else {
      System.setProperty(MAX_CONN_PROPERTY, "16");
    }
  }

  private static void setConfiguration(int maxConn, int beamWidth) {
    System.setProperty(MAX_CONN_PROPERTY, Integer.toString(maxConn));
    System.setProperty(BEAM_WIDTH_PROPERTY, Integer.toString(beamWidth));
  }

  private static void restoreProperty(String name, String value) {
    if (value == null) {
      System.clearProperty(name);
    } else {
      System.setProperty(name, value);
    }
  }
}

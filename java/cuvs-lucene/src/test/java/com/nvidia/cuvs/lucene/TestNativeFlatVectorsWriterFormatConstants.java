/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.junit.Assert.assertEquals;

import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.junit.Test;

/**
 * Coarse tripwire for {@link NativeFlatVectorsWriter}'s hand-transcribed {@code Lucene99} format.
 *
 * <p>{@link NativeFlatVectorsWriter}'s class javadoc pins its dense float32 layout (format
 * constants, header, meta field order, footer) to the frozen {@code Lucene99FlatVectorsFormat}
 * codec, which stays byte-for-byte stable for the life of the codec. {@code Lucene99} support in
 * {@code lucene-backward-codecs} is guaranteed for the current Lucene major version, so this test
 * asserts that the resolved {@code lucene-core} major version (its manifest {@code
 * Specification-Version}, not a parse of {@code pom.xml}'s text) still matches the major version
 * this class was verified against — the signal to re-verify {@code Lucene99} readability on a
 * major Lucene upgrade.
 *
 * <p>{@code TestNativeFlatVectorsWriterRoundTrip} is the positive verification that {@link
 * NativeFlatVectorsWriter} still writes bytes the stock {@code Lucene99FlatVectorsReader} on the
 * classpath can read; run it on every {@code lucene-core} version bump, major or not.
 */
public class TestNativeFlatVectorsWriterFormatConstants {

  private static final String PINNED_LUCENE_MAJOR_VERSION = "10";

  @Test
  public void lucenePinMatchesResolvedClasspathMajorVersion() {
    String resolved = Lucene99FlatVectorsFormat.class.getPackage().getSpecificationVersion();
    String resolvedMajor = resolved.split("\\.", 2)[0];
    assertEquals(
        "lucene-core on the classpath is "
            + resolved
            + ", a new major version from the one ("
            + PINNED_LUCENE_MAJOR_VERSION
            + ") NativeFlatVectorsWriter's Lucene99-format transcription was verified against (per"
            + " its class javadoc). Run TestNativeFlatVectorsWriterRoundTrip to check whether"
            + " Lucene99FlatVectorsReader on the new classpath still reads what"
            + " NativeFlatVectorsWriter writes. If it does, bump PINNED_LUCENE_MAJOR_VERSION here."
            + " If it doesn't, re-port NativeFlatVectorsWriter to the new default flat-vector"
            + " format, then bump PINNED_LUCENE_MAJOR_VERSION.",
        PINNED_LUCENE_MAJOR_VERSION,
        resolvedMajor);
  }
}

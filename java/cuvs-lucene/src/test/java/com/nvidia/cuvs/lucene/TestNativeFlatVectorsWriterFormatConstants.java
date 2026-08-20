/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.junit.Assert.assertEquals;

import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.junit.Test;

/**
 * Tripwire for {@link NativeFlatVectorsWriter}'s hand-transcribed Lucene 10.2.0 format.
 *
 * <p>{@link NativeFlatVectorsWriter}'s class javadoc says its dense float32 layout (format
 * constants, header, meta field order, footer) is transcribed from Lucene 10.2.0 and "must stay
 * equal to the lucene-core version in pom.xml." This asserts that invariant against the resolved
 * {@code lucene-core} jar actually on the classpath (its manifest {@code Specification-Version},
 * not a parse of {@code pom.xml}'s text), so a Lucene upgrade fails this test immediately rather
 * than silently writing {@code .vec}/{@code .vemf} files the stock
 * {@code Lucene99FlatVectorsReader} may no longer read correctly.
 *
 * <p>This intentionally fires on <em>any</em> version change, not just ones that actually alter
 * the format — per the class javadoc, every upgrade needs a manual re-verification pass against
 * the new version's {@code Lucene99FlatVectorsFormat}/{@code Lucene99FlatVectorsWriter} sources.
 */
public class TestNativeFlatVectorsWriterFormatConstants {

  private static final String PINNED_LUCENE_VERSION = "10.2.0";

  @Test
  public void lucenePinMatchesResolvedClasspathVersion() {
    String resolved = Lucene99FlatVectorsFormat.class.getPackage().getSpecificationVersion();
    assertEquals(
        "lucene-core on the classpath is "
            + resolved
            + ", but NativeFlatVectorsWriter is pinned to "
            + PINNED_LUCENE_VERSION
            + " (per its class javadoc). Re-verify NativeFlatVectorsWriter against the new"
            + " version's Lucene99FlatVectorsFormat/Lucene99FlatVectorsWriter sources, update the"
            + " mirrored constants and write sequence if needed, then bump PINNED_LUCENE_VERSION"
            + " here.",
        PINNED_LUCENE_VERSION,
        resolved);
  }
}

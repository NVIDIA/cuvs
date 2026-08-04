/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

import org.apache.lucene.codecs.Codec;
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
    LuceneProvider provider = LuceneProvider.getInstance("99");
    assertTrue(provider.getLuceneFlatVectorsFormatInstance(null) instanceof FlatVectorsFormat);
    assertEquals(provider.getStaticIntParam("VERSION_CURRENT"), 0);
    assertNotEquals(provider.getSimilarityFunctions().size(), 0);
  }
}

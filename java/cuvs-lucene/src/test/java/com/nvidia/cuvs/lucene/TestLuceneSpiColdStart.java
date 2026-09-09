/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import org.junit.Test;

/** Verifies Lucene SPI discovery and format construction with fresh process-local state. */
public class TestLuceneSpiColdStart {

  @Test
  public void testCodecSpiResolvesExpectedCuvsProvidersInFreshJvm() throws Exception {
    SPIColdStartProbe.assertSucceeds(SPIColdStartProbe.CODEC_SPI_DISCOVERY_MODE);
  }

  @Test
  public void testKnnVectorsFormatSpiResolvesExpectedCuvsProvidersInFreshJvm() throws Exception {
    SPIColdStartProbe.assertSucceeds(SPIColdStartProbe.VECTOR_FORMAT_SPI_DISCOVERY_MODE);
  }

  @Test
  public void testHnswFormatConstructionDoesNotResolveLuceneProvider() throws Exception {
    SPIColdStartProbe.assertSucceeds(SPIColdStartProbe.HNSW_CONSTRUCTOR_MODE);
  }

  @Test
  public void testCagraFormatConstructionDoesNotResolveLuceneProvider() throws Exception {
    SPIColdStartProbe.assertSucceeds(SPIColdStartProbe.CAGRA_CONSTRUCTOR_MODE);
  }

  @Test
  public void testScalarFormatConstructionDoesNotResolveLuceneProvider() throws Exception {
    SPIColdStartProbe.assertSucceeds(SPIColdStartProbe.SCALAR_CONSTRUCTOR_MODE);
  }

  @Test
  public void testBinaryFormatConstructionDoesNotResolveLuceneProvider() throws Exception {
    SPIColdStartProbe.assertSucceeds(SPIColdStartProbe.BINARY_CONSTRUCTOR_MODE);
  }
}

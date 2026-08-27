/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CagraSearchParams;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.Test;

@SuppressSysoutChecks(bugUrl = "")
public class TestGPUKnnFloatVectorQueryParameters extends LuceneTestCase {

  private static final float[] TARGET = {0.0f};

  @Test
  public void testRejectsInvalidSearchParameters() {
    assertThrows(
        IllegalArgumentException.class,
        () -> new GPUKnnFloatVectorQuery("vector", TARGET, 1, null, 0, 1));
    assertThrows(
        IllegalArgumentException.class,
        () -> new GPUKnnFloatVectorQuery("vector", TARGET, 1, null, 1, 0));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            new GPUKnnFloatVectorQuery(
                "vector", TARGET, 1, null, 1, GPUKnnFloatVectorQuery.MAX_SEARCH_WIDTH + 1));
  }

  @Test
  public void testAcceptsUniversalSearchParameterBoundaries() {
    new GPUKnnFloatVectorQuery(
        "vector",
        TARGET,
        1,
        null,
        GPUKnnFloatVectorQuery.MIN_ITOPK,
        GPUKnnFloatVectorQuery.MIN_SEARCH_WIDTH);
    new GPUKnnFloatVectorQuery(
        "vector",
        TARGET,
        1,
        null,
        GPUKnnFloatVectorQuery.MAX_ITOPK,
        GPUKnnFloatVectorQuery.MAX_SEARCH_WIDTH,
        0,
        0,
        CagraSearchParams.SearchAlgo.AUTO);
  }

  @Test
  public void testSingleCtaITopKBoundary() {
    new GPUKnnFloatVectorQuery(
        "vector",
        TARGET,
        GPUKnnFloatVectorQuery.MAX_SINGLE_CTA_ITOPK,
        null,
        GPUKnnFloatVectorQuery.MAX_SINGLE_CTA_ITOPK,
        GPUKnnFloatVectorQuery.MIN_SEARCH_WIDTH,
        0,
        0,
        CagraSearchParams.SearchAlgo.SINGLE_CTA);

    assertThrows(
        IllegalArgumentException.class,
        () ->
            new GPUKnnFloatVectorQuery(
                "vector",
                TARGET,
                1,
                null,
                GPUKnnFloatVectorQuery.MAX_SINGLE_CTA_ITOPK + 1,
                GPUKnnFloatVectorQuery.MIN_SEARCH_WIDTH,
                0,
                0,
                CagraSearchParams.SearchAlgo.SINGLE_CTA));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            new GPUKnnFloatVectorQuery(
                "vector",
                TARGET,
                GPUKnnFloatVectorQuery.MAX_SINGLE_CTA_ITOPK + 1,
                null,
                GPUKnnFloatVectorQuery.MAX_SINGLE_CTA_ITOPK,
                GPUKnnFloatVectorQuery.MIN_SEARCH_WIDTH,
                0,
                0,
                CagraSearchParams.SearchAlgo.SINGLE_CTA));
  }
}

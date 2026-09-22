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
  public void testIterationBoundaries() {
    for (int iterations : new int[] {Integer.MIN_VALUE, -1}) {
      IllegalArgumentException e =
          expectThrows(
              IllegalArgumentException.class,
              () ->
                  new GPUKnnFloatVectorQuery(
                      "vector",
                      TARGET,
                      1,
                      null,
                      1,
                      1,
                      0,
                      iterations,
                      CagraSearchParams.SearchAlgo.AUTO));
      assertTrue(e.getMessage(), e.getMessage().contains("maxIterations"));
    }
    // Construction checks representability, not the native plan's memory requirements.
    for (int iterations : new int[] {0, 1, Integer.MAX_VALUE}) {
      new GPUKnnFloatVectorQuery(
          "vector", TARGET, 1, null, 1, 1, 0, iterations, CagraSearchParams.SearchAlgo.AUTO);
    }
  }

  @Test
  public void testThreadBlockSizes() {
    for (int size : new int[] {0, 64, 128, 256, 512, 1024}) {
      new GPUKnnFloatVectorQuery(
          "vector", TARGET, 1, null, 1, 1, size, 0, CagraSearchParams.SearchAlgo.AUTO);
    }
    for (int size :
        new int[] {Integer.MIN_VALUE, -1, 1, 32, 63, 65, 192, 1025, Integer.MAX_VALUE}) {
      IllegalArgumentException e =
          expectThrows(
              IllegalArgumentException.class,
              () ->
                  new GPUKnnFloatVectorQuery(
                      "vector", TARGET, 1, null, 1, 1, size, 0, CagraSearchParams.SearchAlgo.AUTO));
      assertTrue(e.getMessage(), e.getMessage().contains("threadBlockSize"));
    }
  }

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

  /**
   * Verifies only that the Java-level range checks in {@link
   * GPUKnnFloatVectorQuery#MIN_ITOPK}/{@link GPUKnnFloatVectorQuery#MAX_ITOPK} and {@link
   * GPUKnnFloatVectorQuery#MIN_SEARCH_WIDTH}/{@link GPUKnnFloatVectorQuery#MAX_SEARCH_WIDTH}
   * accept their own boundary values at construction time.
   *
   * <p>This does NOT prove these values are supported by native CAGRA: constructing a {@link
   * GPUKnnFloatVectorQuery} never builds a native search plan, and (as documented on {@link
   * GPUKnnFloatVectorQuery#MAX_ITOPK}) native CAGRA sizes internal traversal hash tables from
   * itopk_size, search_width, max_iterations, graph degree, and dataset size — values this test
   * does not exercise. A combination that passes this test can still be rejected by native CAGRA.
   */
  @Test
  public void testAcceptsJavaLevelSearchParameterBoundaries() {
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

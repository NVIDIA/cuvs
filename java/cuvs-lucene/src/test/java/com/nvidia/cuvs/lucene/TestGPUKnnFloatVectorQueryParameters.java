/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

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
    new GPUKnnFloatVectorQuery("vector", TARGET, 1, null, 1, 1);
  }
}

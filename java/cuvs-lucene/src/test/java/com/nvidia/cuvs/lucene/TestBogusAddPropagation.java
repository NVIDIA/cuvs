/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.Test;

/**
 * Verifies end-to-end build propagation from the cuVS C layer, through the cuvs-java bindings, to
 * cuvs-lucene: the bogus {@code cuvsBogusAdd} C function (added in {@code c/src/core/c_api.cpp}) is
 * exposed as {@code CuVSProvider.bogusAdd} in cuvs-java and is called from the cuvs-lucene {@link
 * BogusPropagation#bogusAdd(int, int)} method exercised here. If any layer of the build fails to
 * propagate, this test will fail to compile or fail at runtime.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestBogusAddPropagation extends LuceneTestCase {

  @Test
  public void testBogusAddReachableFromLucene() {
    assertEquals(5, BogusPropagation.bogusAdd(2, 3));
  }
}

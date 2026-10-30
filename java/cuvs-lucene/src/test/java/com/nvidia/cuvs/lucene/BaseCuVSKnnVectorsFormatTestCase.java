/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;

import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.tests.index.BaseKnnVectorsFormatTestCase;
import org.junit.BeforeClass;
import org.junit.Ignore;

/**
 * Runs Lucene's vectors-format test suite against a cuvs-lucene format, adapted to what all of them
 * have in common: they need cuVS, and they index float vectors only.
 *
 * <p>It also absorbs the differences between the versions of Lucene's test suite that cuvs-lucene
 * supports, so that the subclasses compile against all of them.
 */
public abstract class BaseCuVSKnnVectorsFormatTestCase extends BaseKnnVectorsFormatTestCase {

  @BeforeClass
  public static void assumeCuVSSupported() {
    assumeTrue("cuVS is not supported", isSupported());
  }

  @Override
  protected VectorEncoding randomVectorEncoding() {
    return VectorEncoding.FLOAT32;
  }

  // The tests below index byte vectors, which the cuvs-lucene formats do not support.

  @Ignore
  @Override
  public void testByteVectorScorerIteration() {}

  @Ignore
  @Override
  public void testEmptyByteVectorData() {}

  @Ignore
  @Override
  public void testMergingWithDifferentByteKnnFields() {}

  @Ignore
  @Override
  public void testMismatchedFields() {}

  @Ignore
  @Override
  public void testRandomBytes() {}

  @Ignore
  @Override
  public void testSortedIndexBytes() {}
}

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

  // Whether the format can rebuild float vectors from its quantized copy once the raw vectors are
  // gone. Lucene's suite runs its raw-vector fallback tests only when this returns true; subclasses
  // that return true also implement simulateEmptyRawVectors(Directory).
  // BaseKnnVectorsFormatTestCase
  // declares it abstract since Lucene 10.4 and not at all before, so it cannot be marked @Override.
  protected boolean supportsFloatVectorFallback() {
    return false;
  }

  // The tests below index byte vectors, which the cuvs-lucene formats do not support.

  // Lucene 10.5 added this test. It indexes byte vectors whatever randomVectorEncoding() returns.
  // Not marked @Override, so that it compiles against earlier Lucene releases, where it is simply
  // one more skipped test.
  public void testWriterByteVectorRamEstimate() {
    assumeTrue("byte vectors are not supported", false);
  }

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

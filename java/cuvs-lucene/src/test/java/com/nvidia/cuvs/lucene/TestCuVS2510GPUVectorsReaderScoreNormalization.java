/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

public class TestCuVS2510GPUVectorsReaderScoreNormalization extends LuceneTestCase {

  @Test
  public void testMaximumInnerProductValueUsesLuceneScoreContract() {
    assertEquals(
        13.0f,
        CuVS2510GPUVectorsReader.toLuceneScore(
            VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, 12.0f),
        0.0f);
    assertEquals(
        0.2f,
        CuVS2510GPUVectorsReader.toLuceneScore(
            VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, -4.0f),
        0.0f);
  }

  @Test
  public void testOtherCuVSValuesUseTheirLuceneScoreContracts() {
    assertEquals(
        0.2f,
        CuVS2510GPUVectorsReader.toLuceneScore(VectorSimilarityFunction.EUCLIDEAN, 4.0f),
        0.0f);
    assertEquals(
        0.75f,
        CuVS2510GPUVectorsReader.toLuceneScore(VectorSimilarityFunction.DOT_PRODUCT, 0.5f),
        0.0f);
    assertEquals(
        0.75f, CuVS2510GPUVectorsReader.toLuceneScore(VectorSimilarityFunction.COSINE, 0.5f), 0.0f);
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.junit.Assert.assertArrayEquals;

import java.util.List;
import org.junit.Test;

public class TestScalarQuantization {

  @Test
  public void quantizesNegativeOnlyDimensionsAcrossTheFullRange() {
    List<byte[]> quantized =
        AcceleratedHNSWUtils.quantizeFloatVectorsToScalar(
            List.of(new float[] {-10.0f, -4.0f}, new float[] {-5.0f, -2.0f}));

    assertArrayEquals(new byte[] {0, 0}, quantized.get(0));
    assertArrayEquals(new byte[] {127, 127}, quantized.get(1));
  }

  @Test
  public void quantizesIntermediateValuesMonotonically() {
    List<byte[]> quantized =
        AcceleratedHNSWUtils.quantizeFloatVectorsToScalar(
            List.of(
                new float[] {-1.0f},
                new float[] {-0.5f},
                new float[] {0.0f},
                new float[] {0.5f},
                new float[] {1.0f}));

    assertArrayEquals(new byte[] {0}, quantized.get(0));
    assertArrayEquals(new byte[] {32}, quantized.get(1));
    assertArrayEquals(new byte[] {64}, quantized.get(2));
    assertArrayEquals(new byte[] {95}, quantized.get(3));
    assertArrayEquals(new byte[] {127}, quantized.get(4));
  }
}

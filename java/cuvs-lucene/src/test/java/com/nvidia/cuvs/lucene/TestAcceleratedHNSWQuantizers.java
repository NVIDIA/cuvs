/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.BinaryQuantizer;
import com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.ScalarQuantizer;
import java.util.List;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

public class TestAcceleratedHNSWQuantizers extends LuceneTestCase {

  @Test
  public void testBinaryGoldenBytesAndReusableBufferClearing() {
    float[] low = new float[10];
    float[] high = new float[] {2, 0, 2, 0, 2, 0, 2, 0, 2, 0};
    BinaryQuantizer quantizer = new BinaryQuantizer(10);
    quantizer.add(low);
    quantizer.add(high);
    quantizer.finish();

    byte[] scratch = new byte[] {(byte) 0xff, (byte) 0xff};
    quantizer.quantize(high, scratch);
    assertArrayEquals(new byte[] {0x55, 0x01}, scratch);

    quantizer.quantize(low, scratch);
    assertArrayEquals(new byte[] {0x00, 0x00}, scratch);
    List<byte[]> quantized = AcceleratedHNSWUtils.quantizeFloatVectorsToBinary(List.of(low, high));
    assertArrayEquals(new byte[] {0x00, 0x00}, quantized.get(0));
    assertArrayEquals(new byte[] {0x55, 0x01}, quantized.get(1));
  }

  @Test
  public void testScalarGoldenBytesPreserveExistingNegativeAndConstantBehavior() {
    float[] first = new float[] {-4, -2, 7};
    float[] second = new float[] {-2, -2, 7};
    ScalarQuantizer quantizer = new ScalarQuantizer(3);
    quantizer.add(first);
    quantizer.add(second);
    quantizer.finish();

    byte[] scratch = new byte[3];
    quantizer.quantize(first, scratch);
    assertArrayEquals(new byte[] {-64, -64, 0}, scratch);
    quantizer.quantize(second, scratch);
    assertArrayEquals(new byte[] {0, -64, 0}, scratch);

    List<byte[]> quantized =
        AcceleratedHNSWUtils.quantizeFloatVectorsToScalar(List.of(first, second));
    assertArrayEquals(new byte[] {-64, -64, 0}, quantized.get(0));
    assertArrayEquals(new byte[] {0, -64, 0}, quantized.get(1));
  }
}

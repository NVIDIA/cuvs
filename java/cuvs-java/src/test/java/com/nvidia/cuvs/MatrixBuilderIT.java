/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;
import static org.junit.Assert.assertArrayEquals;

import com.carrotsearch.randomizedtesting.RandomizedRunner;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Matrix builders: contents round-trip, and unwritten parts of a host matrix read as zero. */
@RunWith(RandomizedRunner.class)
public class MatrixBuilderIT extends CuVSTestCase {

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxSupportedArch());
    initializeRandom();
  }

  /** Enough rows to cycle through the device builder's staging buffer several times. */
  @Test
  public void testDeviceBuilderAcrossManyStagingChunks() throws Throwable {
    for (int stride : new int[] {-1, 1030}) {
      int rows = 9000, cols = 1024; // 4 KiB rows: ~36 MB, several 8 MB chunks
      var expected = createFloatMatrix(rows, cols);
      try (var resources = CheckedCuVSResources.create();
          var builder =
              stride > 0
                  ? CuVSMatrix.deviceBuilder(
                      resources, rows, cols, stride, -1, CuVSMatrix.DataType.FLOAT)
                  : CuVSMatrix.deviceBuilder(resources, rows, cols, CuVSMatrix.DataType.FLOAT)) {
        for (var row : expected) builder.addVector(row);
        // Read back with toHost(), which copies with the row stride, rather than getRow().
        try (var matrix = builder.build();
            var host = matrix.toHost()) {
          var actual = new float[cols];
          for (int r = 0; r < rows; ++r) {
            host.getRow(r).toArray(actual);
            assertArrayEquals("stride " + stride + " row " + r, expected[r], actual, 0f);
          }
        }
      }
    }
  }

  /** A device builder closed before build() must leave its resources usable. */
  @Test
  public void testAbandonedDeviceBuilderThenReuseResources() throws Throwable {
    var data = createFloatMatrix(3000, 1024);
    try (var resources = CheckedCuVSResources.create()) {
      try (var abandoned =
          CuVSMatrix.deviceBuilder(resources, 3000, 1024, CuVSMatrix.DataType.FLOAT)) {
        for (int r = 0; r < 2500; ++r) abandoned.addVector(data[r]);
      }
      try (var builder =
          CuVSMatrix.deviceBuilder(resources, 3000, 1024, CuVSMatrix.DataType.FLOAT)) {
        for (var row : data) builder.addVector(row);
        try (var matrix = builder.build()) {
          var actual = new float[1024];
          matrix.getRow(2999).toArray(actual);
          assertArrayEquals(data[2999], actual, 0f);
        }
      }
    }
  }

  /** Rows never added read as zero, padded or not. */
  @Test
  public void testHostBuilderUnfilledRowsAreZero() {
    int rows = 100, cols = 5, filled = 60;
    var data = createFloatMatrix(filled, cols);
    for (int stride : new int[] {-1, 8}) {
      try (var builder =
          stride > 0
              ? CuVSMatrix.hostBuilder(rows, cols, stride, -1, CuVSMatrix.DataType.FLOAT)
              : CuVSMatrix.hostBuilder(rows, cols, CuVSMatrix.DataType.FLOAT)) {
        for (var row : data) builder.addVector(row);
        try (var matrix = builder.build()) {
          var actual = new float[cols];
          for (int r = 0; r < rows; ++r) {
            matrix.getRow(r).toArray(actual);
            assertArrayEquals(
                "stride " + stride + " row " + r,
                r < filled ? data[r] : new float[cols],
                actual,
                0f);
          }
        }
      }
    }
  }
}

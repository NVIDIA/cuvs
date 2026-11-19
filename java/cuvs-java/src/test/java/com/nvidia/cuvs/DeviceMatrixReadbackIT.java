/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import com.carrotsearch.randomizedtesting.RandomizedRunner;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Reading device matrices back to the host: row access, bulk copies and device views. */
@RunWith(RandomizedRunner.class)
public class DeviceMatrixReadbackIT extends CuVSTestCase {

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxSupportedArch());
    initializeRandom();
  }

  /** Rows of a padded device matrix read back through the buffered (pinned) row path. */
  @Test
  public void testStridedDeviceMatrixGetRow() throws Throwable {
    int rows = 3000, cols = 5, stride = 8;
    var expected = createFloatMatrix(rows, cols);
    try (var resources = CheckedCuVSResources.create();
        var builder =
            CuVSMatrix.deviceBuilder(
                resources, rows, cols, stride, -1, CuVSMatrix.DataType.FLOAT)) {
      for (var row : expected) {
        builder.addVector(row);
      }
      try (var matrix = builder.build()) {
        var actual = new float[cols];
        for (int r = 0; r < rows; ++r) {
          matrix.getRow(r).toArray(actual);
          assertArrayEquals("row " + r, expected[r], actual, 0f);
        }
        var all = new float[rows][cols];
        matrix.toArray(all);
        for (int r = 0; r < rows; ++r) {
          assertArrayEquals("row " + r, expected[r], all[r], 0f);
        }
      }
    }
  }

  /** toArray over a graph-shaped int matrix larger than one pinned chunk. */
  @Test
  public void testDeviceIntMatrixToArray() throws Throwable {
    int rows = 250_000, cols = 12;
    try (var resources = CheckedCuVSResources.create();
        var builder = CuVSMatrix.deviceBuilder(resources, rows, cols, CuVSMatrix.DataType.INT)) {
      var row = new int[cols];
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) row[c] = r * cols + c;
        builder.addVector(row);
      }
      try (var matrix = builder.build()) {
        var all = new int[rows][cols];
        matrix.toArray(all);
        for (int r = 0; r < rows; ++r) {
          for (int c = 0; c < cols; ++c) {
            assertEquals(r * cols + c, all[r][c]);
          }
        }
      }
    }
  }

  /** Copying from the device-matrix view returned by toDevice() must fill the target. */
  @Test
  public void testDeviceMatrixViewCopiesIntoTarget() throws Throwable {
    var data = createFloatMatrix(64, 16);
    try (var resources = CheckedCuVSResources.create();
        var source = deviceMatrix(resources, data);
        var target = CuVSMatrix.deviceBuilder(resources, 64, 16, CuVSMatrix.DataType.FLOAT)) {
      try (var targetMatrix = target.build()) {
        source.toDevice(resources).toDevice(targetMatrix, resources);
        var actual = new float[64][16];
        targetMatrix.toArray(actual);
        for (int r = 0; r < data.length; ++r) {
          assertArrayEquals(data[r], actual[r], 0f);
        }
      }
    }
  }

  private static CuVSDeviceMatrix deviceMatrix(CuVSResources resources, float[][] data) {
    var builder =
        CuVSMatrix.deviceBuilder(resources, data.length, data[0].length, CuVSMatrix.DataType.FLOAT);
    for (var row : data) builder.addVector(row);
    return builder.build();
  }
}

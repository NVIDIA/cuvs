/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;
import static org.junit.Assert.assertTrue;

import com.nvidia.cuvs.spi.CuVSProvider;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import org.junit.Before;
import org.junit.Test;

public class MemoryTrackingResourcesIT extends CuVSTestCase {

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxAmd64());
  }

  @Test
  public void writesNonEmptyCsv() throws Throwable {
    Path csv = Files.createTempFile("cuvs-mtrack", ".csv");
    try {
      try (var resources =
          CuVSResources.create(
              CuVSProvider.tempDirectory(), csv, Duration.ofMillis(2))) {

        // Allocate / release a couple of small device buffers so the
        // background CSV reporter has something to report.
        var b1 =
            CuVSMatrix.deviceBuilder(resources, 64, 32, CuVSMatrix.DataType.FLOAT);
        for (int i = 0; i < 64; ++i) {
          b1.addVector(new float[32]);
        }
        try (var m1 = b1.build()) {
          var b2 =
              CuVSMatrix.deviceBuilder(resources, 32, 16, CuVSMatrix.DataType.FLOAT);
          for (int i = 0; i < 32; ++i) {
            b2.addVector(new float[16]);
          }
          try (var m2 = b2.build()) {
            // Allow the background CSV reporter at least a few ticks
            // before the matrices are released and the handle closed.
            Thread.sleep(20);
          }
        }
      }
      // closing the resources flushes the CSV and restores globals
      assertTrue("csv should be non-empty", Files.size(csv) > 0);
    } finally {
      Files.deleteIfExists(csv);
    }
  }
}

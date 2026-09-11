/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;

import com.nvidia.cuvs.spi.CuVSProvider;
import org.junit.Assume;

/** Keeps local no-GPU tests skippable while making the required GPU CI job fail closed. */
final class CuVSTestSupport {

  private static final String REQUIRE_GPU_ENV = "CUVS_LUCENE_REQUIRE_GPU_TESTS";

  private CuVSTestSupport() {}

  static void enableRmmOrSkip() {
    try {
      CuVSProvider.provider().enableRMMAsyncMemory();
    } catch (UnsupportedOperationException unsupported) {
      requireOrSkip("cuVS RMM initialization failed: " + unsupported.getMessage(), false);
    }
  }

  static void requireCuvsOrSkip() {
    requireOrSkip("cuVS Java/native support is unavailable", isSupported());
  }

  private static void requireOrSkip(String message, boolean condition) {
    if (!condition && "1".equals(System.getenv(REQUIRE_GPU_ENV))) {
      throw new AssertionError(message + " (required because " + REQUIRE_GPU_ENV + "=1)");
    }
    Assume.assumeTrue(message, condition);
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.spi.CuVSProvider;

/**
 * Bogus helper used to verify end-to-end build propagation from the cuVS C layer, through the
 * cuvs-java bindings, to cuvs-lucene. It delegates to {@link CuVSProvider#bogusAdd(int, int)}, which
 * in turn calls the {@code cuvsBogusAdd} C function. If any layer of the build fails to propagate,
 * code that references this method will fail to compile or fail at runtime.
 */
public final class BogusPropagation {

  private BogusPropagation() {}

  /**
   * Returns the sum of {@code a} and {@code b} by delegating to the bogus native function exposed
   * through cuvs-java.
   *
   * @param a first addend
   * @param b second addend
   * @return the sum {@code a + b}
   */
  public static int bogusAdd(int a, int b) {
    return CuVSProvider.provider().bogusAdd(a, b);
  }
}

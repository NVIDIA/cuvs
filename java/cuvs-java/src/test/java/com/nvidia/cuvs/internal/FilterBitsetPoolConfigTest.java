/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.OptionalLong;
import org.junit.Test;

public class FilterBitsetPoolConfigTest {

  @Test
  public void absentValueUsesDefaultPool() {
    assertPoolBytes(
        FilterBitsetPoolConfig.DEFAULT_FILTER_POOL_BYTES,
        FilterBitsetPoolConfig.resolvePoolBytes(null));
  }

  @Test
  public void zeroExplicitlyDisablesPool() {
    assertFalse(FilterBitsetPoolConfig.resolvePoolBytes("0").isPresent());
    assertFalse(FilterBitsetPoolConfig.resolvePoolBytes(" 0 ").isPresent());
  }

  @Test
  public void positiveValueIsAlignedUp() {
    assertPoolBytes(256L, FilterBitsetPoolConfig.resolvePoolBytes("1"));
    assertPoolBytes(256L, FilterBitsetPoolConfig.resolvePoolBytes("256"));
    assertPoolBytes(512L, FilterBitsetPoolConfig.resolvePoolBytes("257"));
  }

  @Test
  public void invalidValueUsesDefaultPool() {
    assertPoolBytes(
        FilterBitsetPoolConfig.DEFAULT_FILTER_POOL_BYTES,
        FilterBitsetPoolConfig.resolvePoolBytes("-1"));
    assertPoolBytes(
        FilterBitsetPoolConfig.DEFAULT_FILTER_POOL_BYTES,
        FilterBitsetPoolConfig.resolvePoolBytes(""));
    assertPoolBytes(
        FilterBitsetPoolConfig.DEFAULT_FILTER_POOL_BYTES,
        FilterBitsetPoolConfig.resolvePoolBytes("not-a-size"));
  }

  @Test
  public void roundingOverflowUsesDefaultPool() {
    assertPoolBytes(
        FilterBitsetPoolConfig.DEFAULT_FILTER_POOL_BYTES,
        FilterBitsetPoolConfig.resolvePoolBytes(Long.toString(Long.MAX_VALUE)));
  }

  private static void assertPoolBytes(long expected, OptionalLong actual) {
    assertTrue(actual.isPresent());
    assertEquals(expected, actual.getAsLong());
  }
}

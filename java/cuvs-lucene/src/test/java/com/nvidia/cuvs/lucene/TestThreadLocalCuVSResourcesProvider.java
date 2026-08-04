/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

/** GPU-free tests for thread-local cuVS resource configuration. */
public class TestThreadLocalCuVSResourcesProvider {

  @Test
  public void absentOrZeroValueDisablesPool() {
    assertEquals(0, ThreadLocalCuVSResourcesProvider.resolveWorkspacePoolBytes(null));
    assertEquals(0, ThreadLocalCuVSResourcesProvider.resolveWorkspacePoolBytes("0"));
    assertEquals(0, ThreadLocalCuVSResourcesProvider.resolveWorkspacePoolBytes(" 0 "));
  }

  @Test
  public void positiveValueIsAlignedUp() {
    assertEquals(256, ThreadLocalCuVSResourcesProvider.resolveWorkspacePoolBytes("1"));
    assertEquals(256, ThreadLocalCuVSResourcesProvider.resolveWorkspacePoolBytes("256"));
    assertEquals(512, ThreadLocalCuVSResourcesProvider.resolveWorkspacePoolBytes("257"));
  }

  @Test
  public void malformedValueDisablesPool() {
    assertEquals(0, ThreadLocalCuVSResourcesProvider.resolveWorkspacePoolBytes(""));
    assertEquals(0, ThreadLocalCuVSResourcesProvider.resolveWorkspacePoolBytes("1024M"));
  }

  @Test
  public void negativeValueDisablesPool() {
    assertEquals(0, ThreadLocalCuVSResourcesProvider.resolveWorkspacePoolBytes("-1"));
  }

  @Test
  public void roundingOverflowDisablesPool() {
    assertEquals(
        0,
        ThreadLocalCuVSResourcesProvider.resolveWorkspacePoolBytes(
            Long.toString(Long.MAX_VALUE)));
  }
}

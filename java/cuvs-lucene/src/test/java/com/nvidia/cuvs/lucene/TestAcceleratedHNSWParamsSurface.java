/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.junit.Assert.assertFalse;

import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import org.junit.Test;

/**
 * Regression guard for the native-flat-buffering surface narrowing: {@code
 * AcceleratedHNSWParams.Builder#withNumInputVectors} must stay package-private, reachable only
 * from within {@code com.nvidia.cuvs.lucene} (i.e. only by {@link CagraHnswBulkIndexWriter}), not from
 * a generic external Lucene codec user. A future change that accidentally re-widens this method to
 * {@code public} would reopen exactly the footgun this package's design is meant to close, without
 * necessarily being caught by any functional test -- this test exists to catch that specific
 * mistake directly.
 */
public class TestAcceleratedHNSWParamsSurface {

  @Test
  public void testWithNumInputVectorsIsNotPublic() throws NoSuchMethodException {
    Method method =
        AcceleratedHNSWParams.Builder.class.getDeclaredMethod("withNumInputVectors", int.class);
    assertFalse(
        "AcceleratedHNSWParams.Builder#withNumInputVectors must not be public; it should only be"
            + " reachable from within com.nvidia.cuvs.lucene (CagraHnswBulkIndexWriter)",
        Modifier.isPublic(method.getModifiers()));
  }
}

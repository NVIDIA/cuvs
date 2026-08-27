/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

public class TestUtilsThrowableHandling extends LuceneTestCase {

  @Test
  public void testHandleThrowableRethrowsErrorUnchanged() {
    Error error = new AssertionError("fatal failure");

    Error thrown = assertThrows(Error.class, () -> Utils.handleThrowable(error));

    assertSame(error, thrown);
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

public class TestUtilsThrowableHandling extends LuceneTestCase {

  @Test
  public void testHandleThrowableRethrowsIOExceptionUnchanged() {
    IOException exception = new IOException("I/O failure");

    IOException thrown = assertThrows(IOException.class, () -> Utils.handleThrowable(exception));

    assertSame(exception, thrown);
  }

  @Test
  public void testHandleThrowableRethrowsRuntimeExceptionUnchanged() {
    RuntimeException exception = new IllegalStateException("runtime failure");

    RuntimeException thrown =
        assertThrows(RuntimeException.class, () -> Utils.handleThrowable(exception));

    assertSame(exception, thrown);
  }

  @Test
  public void testHandleThrowableRethrowsErrorUnchanged() {
    Error error = new AssertionError("fatal failure");

    Error thrown = assertThrows(Error.class, () -> Utils.handleThrowable(error));

    assertSame(error, thrown);
  }

  @Test
  public void testHandleThrowableWrapsCheckedExceptionWithCause() {
    Exception exception = new Exception("checked failure");

    RuntimeException thrown =
        assertThrows(RuntimeException.class, () -> Utils.handleThrowable(exception));

    assertSame(exception, thrown.getCause());
  }
}

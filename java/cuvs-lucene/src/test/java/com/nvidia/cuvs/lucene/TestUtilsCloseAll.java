/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

public class TestUtilsCloseAll extends LuceneTestCase {

  private static AutoCloseable recording(List<String> closed, String name, Throwable failure) {
    return () -> {
      closed.add(name);
      if (failure instanceof Exception e) throw e;
      if (failure instanceof Error e) throw e;
    };
  }

  @Test
  public void testClosesEverythingAndReturnsNullWhenNothingFails() {
    List<String> closed = new ArrayList<>();

    Throwable result =
        Utils.closeAll(null, recording(closed, "a", null), null, recording(closed, "b", null));

    assertNull(result);
    assertEquals(List.of("a", "b"), closed);
  }

  @Test
  public void testKeepsClosingAfterAFailureAndReturnsTheFirstOne() {
    List<String> closed = new ArrayList<>();
    Exception first = new IllegalStateException("first");
    Exception second = new IllegalArgumentException("second");

    Throwable result =
        Utils.closeAll(
            null,
            recording(closed, "a", first),
            recording(closed, "b", null),
            recording(closed, "c", second));

    assertSame(first, result);
    assertEquals(List.of("a", "b", "c"), closed);
    assertArrayEquals(new Throwable[] {second}, first.getSuppressed());
  }

  @Test
  public void testAttachesCloseFailuresToTheFailureInFlight() {
    List<String> closed = new ArrayList<>();
    Error inFlight = new AssertionError("in flight");
    Exception closeFailure = new IllegalStateException("close");

    Throwable result = Utils.closeAll(inFlight, recording(closed, "a", closeFailure));

    assertSame(inFlight, result);
    assertArrayEquals(new Throwable[] {closeFailure}, inFlight.getSuppressed());
  }

  @Test
  public void testDoesNotSuppressAFailureIntoItself() {
    Exception failure = new IllegalStateException("same");

    Throwable result = Utils.closeAll(failure, recording(new ArrayList<>(), "a", failure));

    assertSame(failure, result);
    assertEquals(0, failure.getSuppressed().length);
  }

  @Test
  public void testRestoresTheInterruptFlagWhenCloseIsInterrupted() {
    try {
      Throwable result =
          Utils.closeAll(
              null,
              () -> {
                throw new InterruptedException("interrupted");
              });

      assertTrue(result instanceof InterruptedException);
      assertTrue(Thread.currentThread().isInterrupted());
    } finally {
      Thread.interrupted();
    }
  }
}

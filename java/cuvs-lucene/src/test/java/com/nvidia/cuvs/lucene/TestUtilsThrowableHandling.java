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

  @Test
  public void testOwnedIndexClosesUntransferredDatasetOnce() throws Exception {
    TrackingCloseable dataset = new TrackingCloseable(null);
    Utils.OwnedIndex<TrackingCloseable> owned = Utils.ownDataset(dataset);

    owned.close();
    owned.close();

    assertEquals(1, dataset.closeCount);
  }

  @Test
  public void testOwnedIndexDoesNotDirectlyCloseDatasetAfterSuccessfulIndexClose()
      throws Exception {
    TrackingCloseable dataset = new TrackingCloseable(null);
    TrackingCloseable index = new TrackingCloseable(null);
    Utils.OwnedIndex<TrackingCloseable> owned = Utils.ownDataset(dataset);
    owned.transferTo(index);

    owned.close();

    assertEquals(1, index.closeCount);
    assertEquals(0, dataset.closeCount);
  }

  @Test
  public void testOwnedIndexPreservesBodyAndNestedCleanupFailures() {
    IOException bodyFailure = new IOException("body");
    IOException indexCloseFailure = new IOException("index close");
    IOException datasetCloseFailure = new IOException("dataset close");
    TrackingCloseable dataset = new TrackingCloseable(datasetCloseFailure);
    TrackingCloseable index = new TrackingCloseable(indexCloseFailure);

    IOException thrown =
        assertThrows(
            IOException.class,
            () -> {
              try (Utils.OwnedIndex<TrackingCloseable> owned = Utils.ownDataset(dataset)) {
                owned.transferTo(index);
                throw bodyFailure;
              }
            });

    assertSame(bodyFailure, thrown);
    assertEquals(1, thrown.getSuppressed().length);
    assertSame(indexCloseFailure, thrown.getSuppressed()[0]);
    assertEquals(1, indexCloseFailure.getSuppressed().length);
    assertSame(datasetCloseFailure, indexCloseFailure.getSuppressed()[0]);
    assertEquals(1, index.closeCount);
    assertEquals(1, dataset.closeCount);
  }

  private static final class TrackingCloseable implements AutoCloseable {
    private final Exception failure;
    private int closeCount;

    private TrackingCloseable(Exception failure) {
      this.failure = failure;
    }

    @Override
    public void close() throws Exception {
      closeCount++;
      if (failure != null) {
        throw failure;
      }
    }
  }
}

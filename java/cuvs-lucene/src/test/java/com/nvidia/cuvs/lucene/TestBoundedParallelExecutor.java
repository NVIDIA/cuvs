/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.util.concurrent.AbstractExecutorService;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicReference;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

public class TestBoundedParallelExecutor extends LuceneTestCase {

  @Test
  public void testEffectiveWorkersAreBoundedByCpuAndWork() {
    assertEquals(1, BoundedParallelExecutor.effectiveWorkerCount(512, 10, 1));
    assertEquals(1, BoundedParallelExecutor.effectiveWorkerCount(1, 10, 8));
    assertEquals(3, BoundedParallelExecutor.effectiveWorkerCount(512, 3, 8));
    assertEquals(8, BoundedParallelExecutor.effectiveWorkerCount(512, 20, 8));
  }

  @Test
  public void testEveryItemIsVisitedExactlyOnce() throws Exception {
    int workItems = 10_003;
    AtomicIntegerArray visits = new AtomicIntegerArray(workItems);

    try (BoundedParallelExecutor executor = BoundedParallelExecutor.create(512, workItems, 8)) {
      executor.invokeRanges(
          workItems,
          (taskIndex, start, end) -> {
            for (int item = start; item < end; item++) {
              visits.incrementAndGet(item);
            }
          });
    }

    for (int item = 0; item < workItems; item++) {
      assertEquals("item " + item, 1, visits.get(item));
    }
  }

  @Test
  public void testFailureWaitsForSiblingBeforeReturning() throws Exception {
    CountDownLatch bothStarted = new CountDownLatch(2);
    CountDownLatch releaseSibling = new CountDownLatch(1);
    AtomicBoolean siblingFinished = new AtomicBoolean();

    try (BoundedParallelExecutor executor = BoundedParallelExecutor.create(2, 2, 2)) {
      IOException thrown =
          assertThrows(
              IOException.class,
              () ->
                  executor.invokeRanges(
                      2,
                      (taskIndex, start, end) -> {
                        bothStarted.countDown();
                        assertTrue(bothStarted.await(10, TimeUnit.SECONDS));
                        if (taskIndex == 0) {
                          releaseSibling.countDown();
                          throw new IOException("expected failure");
                        }
                        assertTrue(releaseSibling.await(10, TimeUnit.SECONDS));
                        siblingFinished.set(true);
                      }));
      assertEquals("expected failure", thrown.getMessage());
      assertTrue("sibling task was still running", siblingFinished.get());
    }
  }

  @Test
  public void testMultipleWorkerFailuresAreSuppressedInTaskOrder() throws Exception {
    IOException first = new IOException("first");
    IOException second = new IOException("second");

    try (BoundedParallelExecutor executor = BoundedParallelExecutor.create(2, 2, 2)) {
      IOException thrown =
          assertThrows(
              IOException.class,
              () ->
                  executor.invokeRanges(
                      2,
                      (taskIndex, start, end) -> {
                        throw taskIndex == 0 ? first : second;
                      }));
      assertSame(first, thrown);
      assertArrayEquals(new Throwable[] {second}, thrown.getSuppressed());
    }
  }

  @Test
  public void testInterruptionStopsWorkersAndRestoresInterruptStatus() throws Exception {
    CountDownLatch workersStarted = new CountDownLatch(2);
    CountDownLatch workerExited = new CountDownLatch(2);
    CountDownLatch blockWorkers = new CountDownLatch(1);
    AtomicReference<Throwable> result = new AtomicReference<>();
    AtomicBoolean interruptRestored = new AtomicBoolean();

    Thread caller =
        new Thread(
            () -> {
              try (BoundedParallelExecutor executor = BoundedParallelExecutor.create(2, 2, 2)) {
                executor.invokeRanges(
                    2,
                    (taskIndex, start, end) -> {
                      workersStarted.countDown();
                      try {
                        blockWorkers.await();
                      } finally {
                        workerExited.countDown();
                      }
                    });
                result.set(new AssertionError("parallel invocation unexpectedly completed"));
              } catch (Throwable failure) {
                result.set(failure);
                interruptRestored.set(Thread.currentThread().isInterrupted());
              }
            },
            "bounded-parallel-interruption-test");

    caller.start();
    try {
      assertTrue(workersStarted.await(10, TimeUnit.SECONDS));
      caller.interrupt();
      caller.join(TimeUnit.SECONDS.toMillis(10));
      assertFalse("caller did not terminate", caller.isAlive());
      assertTrue(result.get() instanceof IOException);
      assertTrue("caller interrupt status was not restored", interruptRestored.get());
      assertEquals("worker remained alive after interruption", 0L, workerExited.getCount());
    } finally {
      blockWorkers.countDown();
      caller.interrupt();
      caller.join(TimeUnit.SECONDS.toMillis(10));
    }
  }

  @Test
  public void testPartialSubmissionRejectionStopsAcceptedWorker() throws Exception {
    CountDownLatch workerStarted = new CountDownLatch(1);
    CountDownLatch workerExited = new CountDownLatch(1);
    ExecutorService rejectingExecutor = new RejectAfterFirstExecutor(workerStarted);

    try (BoundedParallelExecutor executor = new BoundedParallelExecutor(2, rejectingExecutor)) {
      assertThrows(
          RejectedExecutionException.class,
          () ->
              executor.invokeRanges(
                  2,
                  (taskIndex, start, end) -> {
                    workerStarted.countDown();
                    try {
                      new CountDownLatch(1).await();
                    } finally {
                      workerExited.countDown();
                    }
                  }));
      assertEquals("accepted worker remained alive", 0L, workerExited.getCount());
    }
  }

  private static final class RejectAfterFirstExecutor extends AbstractExecutorService {
    private final ExecutorService delegate = Executors.newSingleThreadExecutor();
    private final CountDownLatch firstTaskStarted;
    private int submissions;

    RejectAfterFirstExecutor(CountDownLatch firstTaskStarted) {
      this.firstTaskStarted = firstTaskStarted;
    }

    @Override
    public void execute(Runnable command) {
      if (submissions++ > 0) {
        try {
          if (!firstTaskStarted.await(10, TimeUnit.SECONDS)) {
            throw new AssertionError("accepted task did not start");
          }
        } catch (InterruptedException interrupted) {
          Thread.currentThread().interrupt();
          throw new RejectedExecutionException(
              "interrupted before expected rejection", interrupted);
        }
        throw new RejectedExecutionException("expected rejection");
      }
      delegate.execute(command);
    }

    @Override
    public void shutdown() {
      delegate.shutdown();
    }

    @Override
    public java.util.List<Runnable> shutdownNow() {
      return delegate.shutdownNow();
    }

    @Override
    public boolean isShutdown() {
      return delegate.isShutdown();
    }

    @Override
    public boolean isTerminated() {
      return delegate.isTerminated();
    }

    @Override
    public boolean awaitTermination(long timeout, TimeUnit unit) throws InterruptedException {
      return delegate.awaitTermination(timeout, unit);
    }
  }
}

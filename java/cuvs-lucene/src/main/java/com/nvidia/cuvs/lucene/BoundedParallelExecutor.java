/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/** Runs contiguous work ranges with a bounded number of short-lived platform threads. */
final class BoundedParallelExecutor implements AutoCloseable {

  @FunctionalInterface
  interface RangeTask {
    void run(int taskIndex, int startInclusive, int endExclusive) throws Exception;
  }

  private final int workers;
  private final ExecutorService executor;
  private boolean closed;

  static BoundedParallelExecutor create(int requestedWorkers, int workItems) {
    return create(requestedWorkers, workItems, Runtime.getRuntime().availableProcessors());
  }

  static BoundedParallelExecutor create(
      int requestedWorkers, int workItems, int availableProcessors) {
    int workers = effectiveWorkerCount(requestedWorkers, workItems, availableProcessors);
    return new BoundedParallelExecutor(workers);
  }

  static int effectiveWorkerCount(int requestedWorkers, int workItems, int availableProcessors) {
    if (requestedWorkers <= 1 || workItems <= 1 || availableProcessors <= 1) {
      return 1;
    }
    return Math.min(requestedWorkers, Math.min(workItems, availableProcessors));
  }

  private BoundedParallelExecutor(int workers) {
    this(workers, workers == 1 ? null : Executors.newFixedThreadPool(workers));
  }

  BoundedParallelExecutor(int workers, ExecutorService executor) {
    if (workers < 1 || (workers == 1) != (executor == null)) {
      throw new IllegalArgumentException(
          "executor must be present exactly when workers exceed one");
    }
    this.workers = workers;
    this.executor = executor;
  }

  int workerCountFor(int workItems) {
    return Math.min(workers, Math.max(1, workItems));
  }

  boolean isParallel() {
    return executor != null;
  }

  void invokeRanges(int workItems, RangeTask task) throws IOException {
    if (closed) {
      throw new IllegalStateException("executor is closed");
    }
    if (workItems < 0) {
      throw new IllegalArgumentException("workItems must not be negative");
    }
    if (workItems == 0) {
      return;
    }

    int taskCount = workerCountFor(workItems);
    if (taskCount == 1) {
      runSerial(task, workItems);
      return;
    }

    List<Callable<Void>> tasks = new ArrayList<>(taskCount);
    for (int taskIndex = 0; taskIndex < taskCount; taskIndex++) {
      int rangeStart = Math.toIntExact((long) taskIndex * workItems / taskCount);
      int rangeEnd = Math.toIntExact((long) (taskIndex + 1) * workItems / taskCount);
      int rangeIndex = taskIndex;
      tasks.add(
          () -> {
            task.run(rangeIndex, rangeStart, rangeEnd);
            return null;
          });
    }

    List<Future<Void>> futures;
    try {
      futures = executor.invokeAll(tasks);
    } catch (InterruptedException interrupted) {
      stopAfterInterruption();
      Thread.currentThread().interrupt();
      throw new IOException("Interrupted while waiting for parallel graph work", interrupted);
    } catch (Throwable submissionFailure) {
      stopAndAwait();
      rethrow(submissionFailure);
      throw new AssertionError("unreachable");
    }

    Throwable failure = null;
    for (Future<Void> future : futures) {
      try {
        future.get();
      } catch (InterruptedException interrupted) {
        stopAfterInterruption();
        Thread.currentThread().interrupt();
        throw new IOException("Interrupted while collecting parallel graph work", interrupted);
      } catch (ExecutionException execution) {
        failure = addFailure(failure, execution.getCause());
      } catch (CancellationException cancelled) {
        failure = addFailure(failure, cancelled);
      }
    }
    rethrow(failure);
  }

  private static void runSerial(RangeTask task, int workItems) throws IOException {
    try {
      task.run(0, 0, workItems);
    } catch (Throwable failure) {
      rethrow(failure);
    }
  }

  private static Throwable addFailure(Throwable primary, Throwable secondary) {
    if (primary == null) {
      return secondary;
    }
    if (primary != secondary) {
      primary.addSuppressed(secondary);
    }
    return primary;
  }

  private static void rethrow(Throwable failure) throws IOException {
    if (failure == null) {
      return;
    }
    if (failure instanceof IOException ioException) {
      throw ioException;
    }
    if (failure instanceof RuntimeException runtimeException) {
      throw runtimeException;
    }
    if (failure instanceof Error error) {
      throw error;
    }
    throw new IOException("Parallel graph work failed", failure);
  }

  private void stopAfterInterruption() {
    stopAndAwait();
    closed = true;
  }

  private void stopAndAwait() {
    if (executor != null) {
      executor.shutdownNow();
      awaitTerminationPreservingInterrupt();
    }
  }

  @Override
  public void close() throws IOException {
    if (closed) {
      return;
    }
    closed = true;
    if (executor == null) {
      return;
    }
    executor.shutdown();
    awaitTerminationPreservingInterrupt();
  }

  private void awaitTerminationPreservingInterrupt() {
    boolean interrupted = Thread.interrupted();
    try {
      while (!executor.isTerminated()) {
        try {
          executor.awaitTermination(1, TimeUnit.SECONDS);
        } catch (InterruptedException ignored) {
          interrupted = true;
          executor.shutdownNow();
        }
      }
    } finally {
      if (interrupted) {
        Thread.currentThread().interrupt();
      }
    }
  }
}

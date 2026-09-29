/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.FutureTask;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.ThreadInterruptedException;

/** Shares a bounded set of helper threads across accelerated-HNSW graph operations. */
final class GraphWorkExecutor {
  static final String THREAD_NAME_PREFIX = "cuvs-hnsw-graph-worker-";

  private static final int MAX_WORKERS =
      defaultMaxWorkers(Runtime.getRuntime().availableProcessors());
  private static final long KEEP_ALIVE_SECONDS = 1;
  private static final AtomicInteger NEXT_THREAD_ID = new AtomicInteger();
  private static final ThreadPoolExecutor EXECUTOR =
      newExecutor(MAX_WORKERS, KEEP_ALIVE_SECONDS, TimeUnit.SECONDS);

  private GraphWorkExecutor() {}

  static void invokeAll(List<Callable<Void>> tasks) throws IOException {
    invokeAll(EXECUTOR, tasks);
  }

  /**
   * Runs one task on the calling thread and offers the others to {@code executor}. Direct handoff
   * and caller execution apply backpressure without retaining graph work in a queue.
   */
  static void invokeAll(Executor executor, List<Callable<Void>> tasks) throws IOException {
    Objects.requireNonNull(executor, "executor");
    Objects.requireNonNull(tasks, "tasks");
    if (tasks.isEmpty()) {
      return;
    }
    AtomicBoolean taskFailed = new AtomicBoolean();
    List<FutureTask<Void>> preparedTasks = new ArrayList<>(tasks.size());
    for (Callable<Void> task : tasks) {
      preparedTasks.add(
          new FutureTask<>(stopAfterFailure(Objects.requireNonNull(task, "task"), taskFailed)));
    }

    List<FutureTask<Void>> acceptedTasks = new ArrayList<>(preparedTasks.size());
    Throwable failure = null;
    for (int i = 0; i < preparedTasks.size(); i++) {
      FutureTask<Void> task = preparedTasks.get(i);
      if (i == preparedTasks.size() - 1) {
        acceptedTasks.add(task);
        task.run();
        break;
      }
      try {
        executor.execute(task);
        acceptedTasks.add(task);
      } catch (RejectedExecutionException rejected) {
        acceptedTasks.add(task);
        task.run();
      } catch (RuntimeException | Error submissionFailure) {
        taskFailed.set(true);
        failure = submissionFailure;
        break;
      }
    }

    failure = collectFailures(acceptedTasks, failure);
    if (failure != null) {
      throw IOUtils.rethrowAlways(failure);
    }
  }

  static ThreadPoolExecutor newExecutor(int maxWorkers, long keepAlive, TimeUnit unit) {
    if (maxWorkers < 1) {
      throw new IllegalArgumentException("maxWorkers must be positive");
    }
    if (keepAlive <= 0) {
      throw new IllegalArgumentException("keepAlive must be positive");
    }
    Objects.requireNonNull(unit, "unit");

    return new ThreadPoolExecutor(
        0,
        maxWorkers,
        keepAlive,
        unit,
        new SynchronousQueue<>(),
        workerThreadFactory(),
        new ThreadPoolExecutor.AbortPolicy());
  }

  static int defaultMaxWorkers(int availableProcessors) {
    return availableProcessors <= 1 ? 1 : availableProcessors - 1;
  }

  private static Callable<Void> stopAfterFailure(Callable<Void> task, AtomicBoolean taskFailed) {
    return () -> {
      if (taskFailed.get()) {
        return null;
      }
      try {
        return task.call();
      } catch (Exception | Error failure) {
        taskFailed.set(true);
        throw failure;
      }
    };
  }

  private static ThreadFactory workerThreadFactory() {
    return task -> {
      Thread worker =
          new Thread(null, task, THREAD_NAME_PREFIX + NEXT_THREAD_ID.incrementAndGet(), 0, false);
      worker.setDaemon(true);
      worker.setPriority(Thread.NORM_PRIORITY);
      worker.setContextClassLoader(null);
      return worker;
    };
  }

  /** Waits for all accepted work before returning, even after failure or interruption. */
  private static Throwable collectFailures(List<FutureTask<Void>> tasks, Throwable failure) {
    boolean interrupted = false;
    for (FutureTask<Void> task : tasks) {
      while (true) {
        try {
          task.get();
          break;
        } catch (InterruptedException interruption) {
          interrupted = true;
          failure = addFailure(failure, new ThreadInterruptedException(interruption));
        } catch (ExecutionException taskFailure) {
          failure = addFailure(failure, taskFailure.getCause());
          break;
        }
      }
    }
    if (interrupted) {
      Thread.currentThread().interrupt();
    }
    return failure;
  }

  private static Throwable addFailure(Throwable failure, Throwable next) {
    if (failure == null) {
      return next;
    }
    if (failure != next) {
      failure.addSuppressed(next);
    }
    return failure;
  }
}

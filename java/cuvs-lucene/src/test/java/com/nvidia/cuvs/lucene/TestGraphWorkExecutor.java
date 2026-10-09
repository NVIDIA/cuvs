/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicReference;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.ThreadInterruptedException;
import org.junit.Test;

/** Behavioral specifications for the shared accelerated-HNSW graph-work scheduler. */
public class TestGraphWorkExecutor extends LuceneTestCase {
  private static final long TIMEOUT_SECONDS = 10;

  @Test
  public void defaultWorkerCapReservesTheCallingThread() {
    assertEquals(1, GraphWorkExecutor.defaultMaxWorkers(1));
    assertEquals(1, GraphWorkExecutor.defaultMaxWorkers(2));
    assertEquals(7, GraphWorkExecutor.defaultMaxWorkers(8));
  }

  @Test
  public void saturationRunsWorkOnTheCallerWithoutLosingIt() throws Exception {
    ThreadPoolExecutor executor = newExecutor(1);
    CountDownLatch helperStarted = new CountDownLatch(1);
    CountDownLatch releaseHelper = new CountDownLatch(1);
    try {
      executor.execute(
          () -> {
            helperStarted.countDown();
            awaitUninterruptibly(releaseHelper);
          });
      assertTrue(helperStarted.await(TIMEOUT_SECONDS, TimeUnit.SECONDS));

      AtomicIntegerArray calls = new AtomicIntegerArray(32);
      Set<Thread> taskThreads = ConcurrentHashMap.newKeySet();
      GraphWorkExecutor.invokeAll(executor, countedTasks(calls, taskThreads));

      assertEquals(Set.of(Thread.currentThread()), taskThreads);
      assertEachCalledOnce(calls);
    } finally {
      releaseHelper.countDown();
      shutdown(executor);
    }
  }

  @Test
  public void concurrentCallersUseTheProductionSharedWorkerBound() throws Exception {
    int workerLimit =
        GraphWorkExecutor.defaultMaxWorkers(Runtime.getRuntime().availableProcessors());
    int callers = Math.min(4, workerLimit + 1);
    int tasksPerCaller = Math.ceilDiv(workerLimit + 1, callers) + 1;
    ConcurrentRun run = runConcurrentWork(callers, tasksPerCaller, workerLimit);

    assertEachCalledOnce(run.calls);
    assertTrue(run.maxActiveHelpers > 0);
    assertTrue(run.maxActiveHelpers <= workerLimit);
  }

  @Test
  public void failuresWaitForStartedWorkPreserveCausesAndSkipPendingWork() throws Exception {
    ThreadPoolExecutor graphExecutor = newExecutor(1);
    ExecutorService callerExecutor = Executors.newSingleThreadExecutor();
    CountDownLatch helperStarted = new CountDownLatch(1);
    CountDownLatch releaseHelper = new CountDownLatch(1);
    CountDownLatch callerTaskFailed = new CountDownLatch(1);
    AtomicInteger pendingCalls = new AtomicInteger();
    IOException helperFailure = new IOException("helper failed");
    IllegalStateException callerFailure = new IllegalStateException("caller failed");
    try {
      Future<Throwable> invocation =
          invokeAsync(
              callerExecutor,
              () ->
                  GraphWorkExecutor.invokeAll(
                      graphExecutor,
                      List.of(
                          () -> {
                            helperStarted.countDown();
                            awaitUninterruptibly(releaseHelper);
                            throw helperFailure;
                          },
                          () -> {
                            assertTrue(helperStarted.await(TIMEOUT_SECONDS, TimeUnit.SECONDS));
                            callerTaskFailed.countDown();
                            throw callerFailure;
                          },
                          () -> {
                            pendingCalls.incrementAndGet();
                            return null;
                          })));

      assertTrue(callerTaskFailed.await(TIMEOUT_SECONDS, TimeUnit.SECONDS));
      assertFalse(invocation.isDone());
      releaseHelper.countDown();

      Throwable thrown = invocation.get(TIMEOUT_SECONDS, TimeUnit.SECONDS);
      assertTrue(thrown == helperFailure || thrown == callerFailure);
      Throwable otherFailure = thrown == helperFailure ? callerFailure : helperFailure;
      assertArrayEquals(new Throwable[] {otherFailure}, thrown.getSuppressed());
      assertEquals(0, pendingCalls.get());
    } finally {
      releaseHelper.countDown();
      shutdown(callerExecutor);
      shutdown(graphExecutor);
    }
  }

  @Test
  public void interruptedCallerWaitsForStartedWorkAndRestoresInterrupt() throws Exception {
    ThreadPoolExecutor executor = newExecutor(1);
    BlockingTask helper = new BlockingTask();
    CountDownLatch callerShareFinished = new CountDownLatch(1);
    AtomicReference<Throwable> failure = new AtomicReference<>();
    AtomicBoolean interruptedOnExit = new AtomicBoolean();
    try {
      Thread caller =
          new Thread(
              () -> {
                try {
                  GraphWorkExecutor.invokeAll(
                      executor,
                      List.of(
                          helper,
                          () -> {
                            helper.awaitStarted();
                            callerShareFinished.countDown();
                            return null;
                          }));
                } catch (Throwable thrown) {
                  failure.set(thrown);
                } finally {
                  interruptedOnExit.set(Thread.currentThread().isInterrupted());
                }
              },
              "graph-work-interrupted-caller");
      caller.start();

      assertTrue(callerShareFinished.await(TIMEOUT_SECONDS, TimeUnit.SECONDS));
      awaitThreadWaiting(caller);
      caller.interrupt();
      awaitInterruptConsumed(caller);
      assertTrue("caller returned while accepted work was still running", caller.isAlive());

      helper.release();
      caller.join(TimeUnit.SECONDS.toMillis(TIMEOUT_SECONDS));
      assertFalse("caller did not finish", caller.isAlive());
      assertTrue(failure.get() instanceof ThreadInterruptedException);
      assertTrue(interruptedOnExit.get());
    } finally {
      helper.release();
      shutdown(executor);
    }
  }

  @Test
  public void submissionFailureWaitsForAcceptedWork() throws Exception {
    ThreadPoolExecutor delegate = newExecutor(1);
    ExecutorService callerExecutor = Executors.newSingleThreadExecutor();
    BlockingTask acceptedTask = new BlockingTask();
    IllegalStateException expected = new IllegalStateException("submission failed");
    CountDownLatch submissionFailed = new CountDownLatch(1);
    AtomicInteger submissions = new AtomicInteger();
    Executor failsAfterFirstSubmission =
        task -> {
          if (submissions.incrementAndGet() == 1) {
            delegate.execute(task);
            acceptedTask.awaitStartedUninterruptibly();
          } else {
            submissionFailed.countDown();
            throw expected;
          }
        };
    try {
      Future<Throwable> invocation =
          invokeAsync(
              callerExecutor,
              () ->
                  GraphWorkExecutor.invokeAll(
                      failsAfterFirstSubmission, List.of(acceptedTask, () -> null, () -> null)));

      assertTrue(submissionFailed.await(TIMEOUT_SECONDS, TimeUnit.SECONDS));
      assertFalse(invocation.isDone());
      acceptedTask.release();

      assertSame(expected, invocation.get(TIMEOUT_SECONDS, TimeUnit.SECONDS));
    } finally {
      acceptedTask.release();
      shutdown(callerExecutor);
      shutdown(delegate);
    }
  }

  @Test
  public void workersAreIsolatedDaemonsThatExpireWhenIdle() throws Exception {
    ThreadPoolExecutor executor = GraphWorkExecutor.newExecutor(1, 25, TimeUnit.MILLISECONDS);
    InheritableThreadLocal<String> callerState = new InheritableThreadLocal<>();
    ClassLoader originalContextLoader = Thread.currentThread().getContextClassLoader();
    ClassLoader callerContextLoader = new ClassLoader(originalContextLoader) {};
    AtomicReference<WorkerObservation> observation = new AtomicReference<>();
    CountDownLatch workerObserved = new CountDownLatch(1);
    try {
      callerState.set("caller-state");
      Thread.currentThread().setContextClassLoader(callerContextLoader);
      Callable<Void> observeWorker =
          () -> {
            if (isGraphWorker()) {
              Thread worker = Thread.currentThread();
              observation.set(
                  new WorkerObservation(worker, callerState.get(), worker.getContextClassLoader()));
              workerObserved.countDown();
            } else {
              assertTrue(workerObserved.await(TIMEOUT_SECONDS, TimeUnit.SECONDS));
            }
            return null;
          };

      GraphWorkExecutor.invokeAll(executor, List.of(observeWorker, observeWorker));

      WorkerObservation first = observation.get();
      assertNotNull(first);
      assertTrue(first.thread.isDaemon());
      assertTrue(first.thread.getName().startsWith(GraphWorkExecutor.THREAD_NAME_PREFIX));
      assertNull(first.inheritedState);
      assertNull(first.contextClassLoader);

      awaitPoolSize(executor, 0);
      Thread replacement = invokeAndCaptureWorker(executor);
      assertNotSame(first.thread, replacement);
    } finally {
      Thread.currentThread().setContextClassLoader(originalContextLoader);
      callerState.remove();
      shutdown(executor);
    }
  }

  private static ConcurrentRun runConcurrentWork(int callers, int tasksPerCaller, int workerLimit)
      throws Exception {
    ExecutorService callerExecutor = Executors.newFixedThreadPool(callers);
    AtomicIntegerArray calls = new AtomicIntegerArray(callers * tasksPerCaller);
    AtomicInteger activeHelpers = new AtomicInteger();
    AtomicInteger maxActiveHelpers = new AtomicInteger();
    CountDownLatch start = new CountDownLatch(1);
    CountDownLatch releaseHelpers = new CountDownLatch(1);
    CountDownLatch sharedCapacityReached = new CountDownLatch(workerLimit);
    List<Future<?>> invocations = new ArrayList<>();
    try {
      for (int caller = 0; caller < callers; caller++) {
        int invocationId = caller;
        invocations.add(
            callerExecutor.submit(
                () -> {
                  assertTrue(start.await(TIMEOUT_SECONDS, TimeUnit.SECONDS));
                  List<Callable<Void>> tasks = new ArrayList<>(tasksPerCaller);
                  for (int task = 0; task < tasksPerCaller; task++) {
                    int taskId = invocationId * tasksPerCaller + task;
                    tasks.add(
                        () -> {
                          boolean helper = isGraphWorker();
                          if (helper) {
                            int active = activeHelpers.incrementAndGet();
                            maxActiveHelpers.accumulateAndGet(active, Math::max);
                            sharedCapacityReached.countDown();
                          }
                          try {
                            calls.incrementAndGet(taskId);
                            if (helper) {
                              assertTrue(releaseHelpers.await(TIMEOUT_SECONDS, TimeUnit.SECONDS));
                            }
                          } finally {
                            if (helper) {
                              activeHelpers.decrementAndGet();
                            }
                          }
                          return null;
                        });
                  }
                  GraphWorkExecutor.invokeAll(tasks);
                  return null;
                }));
      }
      start.countDown();
      assertTrue(sharedCapacityReached.await(TIMEOUT_SECONDS, TimeUnit.SECONDS));
      releaseHelpers.countDown();
      awaitAll(invocations);
      return new ConcurrentRun(calls, maxActiveHelpers.get());
    } finally {
      releaseHelpers.countDown();
      shutdown(callerExecutor);
    }
  }

  private static List<Callable<Void>> countedTasks(
      AtomicIntegerArray calls, Set<Thread> taskThreads) {
    List<Callable<Void>> tasks = new ArrayList<>(calls.length());
    for (int task = 0; task < calls.length(); task++) {
      int taskId = task;
      tasks.add(
          () -> {
            calls.incrementAndGet(taskId);
            taskThreads.add(Thread.currentThread());
            return null;
          });
    }
    return tasks;
  }

  private static Future<Throwable> invokeAsync(
      ExecutorService executor, ThrowingAction invocation) {
    return executor.submit(
        () -> {
          try {
            invocation.run();
            return null;
          } catch (Throwable failure) {
            return failure;
          }
        });
  }

  private static Thread invokeAndCaptureWorker(ThreadPoolExecutor executor) throws IOException {
    AtomicReference<Thread> worker = new AtomicReference<>();
    CountDownLatch observed = new CountDownLatch(1);
    Callable<Void> capture =
        () -> {
          if (isGraphWorker()) {
            worker.set(Thread.currentThread());
            observed.countDown();
          } else {
            assertTrue(observed.await(TIMEOUT_SECONDS, TimeUnit.SECONDS));
          }
          return null;
        };
    GraphWorkExecutor.invokeAll(executor, List.of(capture, capture));
    assertNotNull(worker.get());
    return worker.get();
  }

  private static boolean isGraphWorker() {
    return Thread.currentThread().getName().startsWith(GraphWorkExecutor.THREAD_NAME_PREFIX);
  }

  private static ThreadPoolExecutor newExecutor(int maxWorkers) {
    return GraphWorkExecutor.newExecutor(maxWorkers, 1, TimeUnit.MINUTES);
  }

  private static void assertEachCalledOnce(AtomicIntegerArray calls) {
    for (int task = 0; task < calls.length(); task++) {
      assertEquals("task " + task, 1, calls.get(task));
    }
  }

  private static void awaitAll(List<Future<?>> futures) throws Exception {
    for (Future<?> future : futures) {
      future.get(TIMEOUT_SECONDS, TimeUnit.SECONDS);
    }
  }

  private static void awaitPoolSize(ThreadPoolExecutor executor, int expected)
      throws InterruptedException {
    long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(TIMEOUT_SECONDS);
    while (executor.getPoolSize() != expected && System.nanoTime() < deadline) {
      Thread.sleep(10);
    }
    assertEquals("pool size did not reach " + expected, expected, executor.getPoolSize());
  }

  private static void awaitThreadWaiting(Thread thread) throws InterruptedException {
    long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(TIMEOUT_SECONDS);
    while (thread.getState() != Thread.State.WAITING && System.nanoTime() < deadline) {
      Thread.sleep(1);
    }
    assertEquals("thread did not enter a wait", Thread.State.WAITING, thread.getState());
  }

  private static void awaitInterruptConsumed(Thread thread) throws InterruptedException {
    long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(TIMEOUT_SECONDS);
    while (thread.isInterrupted() && System.nanoTime() < deadline) {
      Thread.sleep(1);
    }
    assertTrue("caller returned before its accepted work", thread.isAlive());
    assertFalse("caller did not consume its interrupt", thread.isInterrupted());
  }

  private static void awaitUninterruptibly(CountDownLatch latch) {
    boolean interrupted = false;
    while (true) {
      try {
        latch.await();
        break;
      } catch (InterruptedException ignored) {
        interrupted = true;
      }
    }
    if (interrupted) {
      Thread.currentThread().interrupt();
    }
  }

  private static void shutdown(ExecutorService executor) throws InterruptedException {
    executor.shutdownNow();
    assertTrue(executor.awaitTermination(TIMEOUT_SECONDS, TimeUnit.SECONDS));
  }

  private static final class BlockingTask implements Callable<Void> {
    private final CountDownLatch started = new CountDownLatch(1);
    private final CountDownLatch release = new CountDownLatch(1);

    @Override
    public Void call() {
      started.countDown();
      awaitUninterruptibly(release);
      return null;
    }

    private void awaitStarted() throws InterruptedException {
      assertTrue(started.await(TIMEOUT_SECONDS, TimeUnit.SECONDS));
    }

    private void awaitStartedUninterruptibly() {
      awaitUninterruptibly(started);
    }

    private void release() {
      release.countDown();
    }
  }

  @FunctionalInterface
  private interface ThrowingAction {
    void run() throws Throwable;
  }

  private record ConcurrentRun(AtomicIntegerArray calls, int maxActiveHelpers) {}

  private record WorkerObservation(
      Thread thread, String inheritedState, ClassLoader contextClassLoader) {}
}

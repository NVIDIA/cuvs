/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.hnsw.NeighborArray;
import org.junit.Test;

/** Behavioral specifications for native graph-copy admission control. */
public class TestGraphCopyMemoryBudget extends LuceneTestCase {
  private static final long TIMEOUT_SECONDS = 10;

  @Test
  public void reservationsFollowEstimatedPeakAcrossSupportedDegrees() {
    int rows = 100;
    for (int degree : new int[] {1, 32, 512}) {
      long required = GraphCopyMemoryBudget.requiredHeadroom(rows, degree);
      long adjacencyBytes = (long) rows * degree * Integer.BYTES;
      long expected =
          2 * adjacencyBytes
              + RamUsageEstimator.shallowSizeOf(new NeighborArray[rows])
              + rows
                  * (RamUsageEstimator.shallowSizeOfInstance(NeighborArray.class)
                      + RamUsageEstimator.sizeOf(new int[degree])
                      + RamUsageEstimator.sizeOf(new float[degree]));
      assertEquals(expected, required);
      assertTrue("object layout must be included", required > 4 * adjacencyBytes);

      GraphCopyMemoryBudget exactBudget = budgetWith(required, required);
      try (GraphCopyMemoryBudget.Reservation ignored = reserve(exactBudget, rows, degree)) {
        assertTrue(exactBudget.tryReserve(1, 1).isEmpty());
      }

      GraphCopyMemoryBudget insufficientBudget = budgetWith(required, required - 1);
      assertTrue(insufficientBudget.tryReserve(rows, degree).isEmpty());
    }
  }

  @Test
  public void concurrentReservationsCannotExceedSharedHeadroom() throws Exception {
    long rows = 100;
    long degree = 16;
    long reservationBytes = GraphCopyMemoryBudget.requiredHeadroom(rows, degree);
    GraphCopyMemoryBudget budget = budgetWith(2 * reservationBytes, 2 * reservationBytes);
    int callers = 8;
    ExecutorService executor = Executors.newFixedThreadPool(callers);
    CountDownLatch start = new CountDownLatch(1);
    CountDownLatch attempted = new CountDownLatch(callers);
    CountDownLatch release = new CountDownLatch(1);
    AtomicInteger granted = new AtomicInteger();
    List<Future<?>> futures = new ArrayList<>();
    try {
      for (int i = 0; i < callers; i++) {
        futures.add(
            executor.submit(
                () -> {
                  assertTrue(start.await(TIMEOUT_SECONDS, TimeUnit.SECONDS));
                  Optional<GraphCopyMemoryBudget.Reservation> reservation =
                      budget.tryReserve(rows, degree);
                  reservation.ifPresent(ignored -> granted.incrementAndGet());
                  attempted.countDown();
                  if (reservation.isPresent()) {
                    assertTrue(release.await(TIMEOUT_SECONDS, TimeUnit.SECONDS));
                    reservation.orElseThrow().close();
                  }
                  return null;
                }));
      }

      start.countDown();
      assertTrue(attempted.await(TIMEOUT_SECONDS, TimeUnit.SECONDS));
      assertEquals(2, granted.get());
      release.countDown();
      for (Future<?> future : futures) {
        future.get(TIMEOUT_SECONDS, TimeUnit.SECONDS);
      }
    } finally {
      release.countDown();
      executor.shutdownNow();
      assertTrue(executor.awaitTermination(TIMEOUT_SECONDS, TimeUnit.SECONDS));
    }
  }

  @Test
  public void reservationIsReleasedOnFailureAndCloseIsIdempotent() {
    long required = GraphCopyMemoryBudget.requiredHeadroom(100, 16);
    GraphCopyMemoryBudget budget = budgetWith(required, required);
    GraphCopyMemoryBudget.Reservation failedOperation = reserve(budget, 100, 16);
    RuntimeException expected = new RuntimeException("expected");

    RuntimeException actual =
        assertThrows(
            RuntimeException.class,
            () -> {
              try (failedOperation) {
                throw expected;
              }
            });
    assertSame(expected, actual);
    failedOperation.close();

    try (GraphCopyMemoryBudget.Reservation replacement = reserve(budget, 100, 16)) {
      assertTrue(budget.tryReserve(1, 1).isEmpty());
    }
  }

  @Test
  public void invalidOrUnavailableMemoryInformationFailsClosed() {
    assertRejected(() -> null);
    assertRejected(() -> new GraphCopyMemoryBudget.MemorySnapshot(0, 0));
    assertRejected(() -> new GraphCopyMemoryBudget.MemorySnapshot(1_000, -1));
    assertRejected(() -> new GraphCopyMemoryBudget.MemorySnapshot(1_000, 1_001));
    assertRejected(
        () -> {
          throw new UnsupportedOperationException("unavailable");
        });

    GraphCopyMemoryBudget budget =
        budgetWith(/* totalBytes= */ Long.MAX_VALUE, /* freeBytes= */ Long.MAX_VALUE);
    assertTrue(budget.tryReserve(Integer.MAX_VALUE, Integer.MAX_VALUE).isEmpty());
    assertTrue(budget.tryReserve(0, 1).isEmpty());
    assertTrue(budget.tryReserve(1, 0).isEmpty());
    assertTrue(budget.tryReserve(-1, 1).isEmpty());
    assertTrue(budget.tryReserve(1, -1).isEmpty());
  }

  private static GraphCopyMemoryBudget budgetWith(long totalBytes, long freeBytes) {
    return new GraphCopyMemoryBudget(
        () -> new GraphCopyMemoryBudget.MemorySnapshot(totalBytes, freeBytes));
  }

  private static GraphCopyMemoryBudget.Reservation reserve(
      GraphCopyMemoryBudget budget, long rows, long columns) {
    return budget.tryReserve(rows, columns).orElseThrow();
  }

  private static void assertRejected(GraphCopyMemoryBudget.MemoryProbe probe) {
    assertTrue(new GraphCopyMemoryBudget(probe).tryReserve(1, 1).isEmpty());
  }
}

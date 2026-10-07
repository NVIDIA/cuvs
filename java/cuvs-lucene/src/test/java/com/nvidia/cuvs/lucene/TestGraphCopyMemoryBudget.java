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
import org.junit.Test;

/** Behavioral specifications for temporary native graph-copy admission control. */
public class TestGraphCopyMemoryBudget extends LuceneTestCase {
  private static final long TIMEOUT_SECONDS = 10;

  @Test
  public void defaultBudgetAdmitsExpectedBenchmarkShapes() {
    long defaultBudget = AcceleratedHNSWParams.DEFAULT_GRAPH_COPY_MEMORY_BUDGET_BYTES;
    long deep100MRequired = GraphCopyMemoryBudget.requiredCopyBytes(100_000_000L, 32);
    long jasper10MRequired = GraphCopyMemoryBudget.requiredCopyBytes(10_000_000L, 32);

    assertEquals(12_800_000_000L, deep100MRequired);
    assertEquals(1_280_000_000L, jasper10MRequired);
    assertTrue(deep100MRequired < defaultBudget);
    assertTrue(jasper10MRequired < defaultBudget);

    GraphCopyMemoryBudget budget = new GraphCopyMemoryBudget();
    try (GraphCopyMemoryBudget.Reservation ignored =
        budget.tryReserve(100_000_000L, 32, defaultBudget).orElseThrow()) {
      // The default admits one 100M-by-32 temporary copy.
    }
  }

  @Test
  public void reservationsFollowRawInt32PayloadAcrossSupportedDegrees() {
    int rows = 100;
    for (int degree : new int[] {1, 32, 512}) {
      long required = (long) rows * degree * Integer.BYTES;
      assertEquals(required, GraphCopyMemoryBudget.requiredCopyBytes(rows, degree));

      GraphCopyMemoryBudget exactBudget = new GraphCopyMemoryBudget();
      try (GraphCopyMemoryBudget.Reservation ignored =
          exactBudget.tryReserve(rows, degree, required).orElseThrow()) {
        assertTrue(exactBudget.tryReserve(1, 1, required).isEmpty());
      }

      assertTrue(new GraphCopyMemoryBudget().tryReserve(rows, degree, required - 1).isEmpty());
    }
  }

  @Test
  public void concurrentReservationsCannotExceedRequestCeiling() throws Exception {
    long rows = 100;
    long degree = 16;
    long reservationBytes = GraphCopyMemoryBudget.requiredCopyBytes(rows, degree);
    GraphCopyMemoryBudget budget = new GraphCopyMemoryBudget();
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
                      budget.tryReserve(rows, degree, 2 * reservationBytes);
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
    long required = GraphCopyMemoryBudget.requiredCopyBytes(100, 16);
    GraphCopyMemoryBudget budget = new GraphCopyMemoryBudget();
    GraphCopyMemoryBudget.Reservation failedOperation =
        budget.tryReserve(100, 16, required).orElseThrow();
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

    try (GraphCopyMemoryBudget.Reservation ignored =
        budget.tryReserve(100, 16, required).orElseThrow()) {
      assertTrue(budget.tryReserve(1, 1, required).isEmpty());
    }
  }

  @Test
  public void invalidShapesAndBudgetsFailClosed() {
    GraphCopyMemoryBudget budget = new GraphCopyMemoryBudget();
    assertTrue(budget.tryReserve(Integer.MAX_VALUE, Integer.MAX_VALUE, Long.MAX_VALUE).isEmpty());
    assertTrue(budget.tryReserve(0, 1, Long.MAX_VALUE).isEmpty());
    assertTrue(budget.tryReserve(1, 0, Long.MAX_VALUE).isEmpty());
    assertTrue(budget.tryReserve(-1, 1, Long.MAX_VALUE).isEmpty());
    assertTrue(budget.tryReserve(1, -1, Long.MAX_VALUE).isEmpty());
    assertTrue(budget.tryReserve(1, 1, -1).isEmpty());
  }
}

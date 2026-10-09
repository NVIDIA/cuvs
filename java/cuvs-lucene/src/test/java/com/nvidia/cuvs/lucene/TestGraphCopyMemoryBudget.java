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
    long deep100MFinalDegree48Required = GraphCopyMemoryBudget.requiredCopyBytes(100_000_000L, 48);
    long deep100MDefaultDegreeRequired = GraphCopyMemoryBudget.requiredCopyBytes(100_000_000L, 64);
    long deep100MAboveDefaultDegreeRequired =
        GraphCopyMemoryBudget.requiredCopyBytes(100_000_000L, 65);
    long jasper10MRequired = GraphCopyMemoryBudget.requiredCopyBytes(10_000_000L, 32);

    assertEquals(24L << 30, defaultBudget);
    assertEquals(12_800_000_000L, deep100MRequired);
    assertEquals(19_200_000_000L, deep100MFinalDegree48Required);
    assertEquals(25_600_000_000L, deep100MDefaultDegreeRequired);
    assertEquals(26_000_000_000L, deep100MAboveDefaultDegreeRequired);
    assertEquals(1_280_000_000L, jasper10MRequired);
    assertTrue(deep100MRequired < defaultBudget);
    assertTrue(deep100MFinalDegree48Required < defaultBudget);
    assertTrue(deep100MDefaultDegreeRequired < defaultBudget);
    assertTrue(deep100MAboveDefaultDegreeRequired > defaultBudget);
    assertTrue(jasper10MRequired < defaultBudget);

    GraphCopyMemoryBudget budget = new GraphCopyMemoryBudget();
    try (GraphCopyMemoryBudget.Reservation ignored =
        budget.tryReserve(100_000_000L, 32, defaultBudget).orElseThrow()) {
      // The default admits one 100M-by-32 temporary copy.
    }
    try (GraphCopyMemoryBudget.Reservation ignored =
        budget.tryReserve(100_000_000L, 64, defaultBudget).orElseThrow()) {
      // The default also admits one 100M graph using the stock degree of 64.
    }
    assertTrue(budget.tryReserve(100_000_000L, 65, defaultBudget).isEmpty());
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
  public void overlappingReservationsCannotMixConfiguredCeilings() {
    long smallCopyBytes = GraphCopyMemoryBudget.requiredCopyBytes(100, 1);
    long largeCopyBytes = GraphCopyMemoryBudget.requiredCopyBytes(300, 1);
    long smallCeiling = 2 * smallCopyBytes;
    long largeCeiling = 2 * largeCopyBytes;
    GraphCopyMemoryBudget budget = new GraphCopyMemoryBudget();

    try (GraphCopyMemoryBudget.Reservation ignored =
        budget.tryReserve(100, 1, smallCeiling).orElseThrow()) {
      assertTrue(budget.tryReserve(300, 1, largeCeiling).isEmpty());
    }
    try (GraphCopyMemoryBudget.Reservation ignored =
        budget.tryReserve(300, 1, largeCeiling).orElseThrow()) {
      assertTrue(budget.tryReserve(100, 1, smallCeiling).isEmpty());
      assertTrue(
          budget
              .tryReserve(100, 1, AcceleratedHNSWParams.UNLIMITED_GRAPH_COPY_MEMORY_BUDGET_BYTES)
              .isEmpty());
    }

    try (GraphCopyMemoryBudget.Reservation first =
            budget
                .tryReserve(100, 1, AcceleratedHNSWParams.UNLIMITED_GRAPH_COPY_MEMORY_BUDGET_BYTES)
                .orElseThrow();
        GraphCopyMemoryBudget.Reservation second =
            budget
                .tryReserve(300, 1, AcceleratedHNSWParams.UNLIMITED_GRAPH_COPY_MEMORY_BUDGET_BYTES)
                .orElseThrow()) {
      assertTrue(budget.tryReserve(100, 1, smallCeiling).isEmpty());
    }

    try (GraphCopyMemoryBudget.Reservation ignored =
        budget.tryReserve(100, 1, smallCeiling).orElseThrow()) {
      // Releasing all unlimited reservations resets the active policy.
    }
  }

  @Test
  public void unlimitedReservationsFailClosedOnAccountingOverflow() {
    long rows = Integer.MAX_VALUE;
    long columns = 1_000_000_000L;
    long requiredCopyBytes = GraphCopyMemoryBudget.requiredCopyBytes(rows, columns);
    assertTrue(requiredCopyBytes > Long.MAX_VALUE / 2);

    GraphCopyMemoryBudget budget = new GraphCopyMemoryBudget();
    try (GraphCopyMemoryBudget.Reservation ignored =
        budget
            .tryReserve(
                rows, columns, AcceleratedHNSWParams.UNLIMITED_GRAPH_COPY_MEMORY_BUDGET_BYTES)
            .orElseThrow()) {
      assertTrue(
          budget
              .tryReserve(
                  rows, columns, AcceleratedHNSWParams.UNLIMITED_GRAPH_COPY_MEMORY_BUDGET_BYTES)
              .isEmpty());
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
    assertTrue(budget.tryReserve(1, 1, -2).isEmpty());
    assertTrue(budget.tryReserve(1, 1, Long.MIN_VALUE).isEmpty());
  }
}

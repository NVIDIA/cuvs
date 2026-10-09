/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.util.Optional;

/** Coordinates temporary native graph copies across concurrent segment flushes. */
final class GraphCopyMemoryBudget {
  private static final long NO_ACTIVE_BUDGET_BYTES = Long.MIN_VALUE;
  private static final GraphCopyMemoryBudget SHARED = new GraphCopyMemoryBudget();

  private long reservedCopyBytes;
  private long activeBudgetBytes = NO_ACTIVE_BUDGET_BYTES;

  static GraphCopyMemoryBudget shared() {
    return SHARED;
  }

  /**
   * Tries to reserve the raw INT32 payload of one temporary device-to-host adjacency copy.
   * Reservations are shared by callers in this class loader. Overlapping reservations must use
   * the same configured budget; a caller with a different budget is denied until the active
   * reservations are released. A budget of {@code -1} is unlimited, but its reservations remain
   * accounted and cannot overlap a finite policy. This keeps one caller from silently replacing
   * another caller's active policy.
   */
  synchronized Optional<Reservation> tryReserve(
      long rows, long columns, long configuredBudgetBytes) {
    long requiredCopyBytes = requiredCopyBytes(rows, columns);
    if (requiredCopyBytes < 0
        || configuredBudgetBytes < AcceleratedHNSWParams.UNLIMITED_GRAPH_COPY_MEMORY_BUDGET_BYTES) {
      return Optional.empty();
    }
    boolean unlimited =
        configuredBudgetBytes == AcceleratedHNSWParams.UNLIMITED_GRAPH_COPY_MEMORY_BUDGET_BYTES;
    if (!unlimited && requiredCopyBytes > configuredBudgetBytes) {
      return Optional.empty();
    }
    if (reservedCopyBytes != 0 && activeBudgetBytes != configuredBudgetBytes) {
      return Optional.empty();
    }
    if (requiredCopyBytes > Long.MAX_VALUE - reservedCopyBytes) {
      return Optional.empty();
    }
    if (!unlimited
        && (reservedCopyBytes > configuredBudgetBytes
            || requiredCopyBytes > configuredBudgetBytes - reservedCopyBytes)) {
      return Optional.empty();
    }
    if (reservedCopyBytes == 0) {
      activeBudgetBytes = configuredBudgetBytes;
    }
    reservedCopyBytes += requiredCopyBytes;
    return Optional.of(new Reservation(this, requiredCopyBytes));
  }

  /** Returns the raw INT32 adjacency payload, or {@code -1} for an invalid/overflowing shape. */
  static long requiredCopyBytes(long rows, long columns) {
    if (rows <= 0 || rows > Integer.MAX_VALUE || columns <= 0 || columns > Integer.MAX_VALUE) {
      return -1;
    }
    try {
      return Math.multiplyExact(Math.multiplyExact(rows, columns), Integer.BYTES);
    } catch (ArithmeticException overflow) {
      return -1;
    }
  }

  private synchronized void release(Reservation reservation) {
    if (reservation.released) {
      return;
    }
    reservedCopyBytes -= reservation.copyBytes;
    if (reservedCopyBytes == 0) {
      activeBudgetBytes = NO_ACTIVE_BUDGET_BYTES;
    }
    reservation.released = true;
  }

  static final class Reservation implements AutoCloseable {
    private final GraphCopyMemoryBudget budget;
    private final long copyBytes;
    private boolean released;

    private Reservation(GraphCopyMemoryBudget budget, long copyBytes) {
      this.budget = budget;
      this.copyBytes = copyBytes;
    }

    @Override
    public void close() {
      budget.release(this);
    }
  }
}

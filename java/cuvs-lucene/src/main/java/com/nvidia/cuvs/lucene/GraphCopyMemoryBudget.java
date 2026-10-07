/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.util.Optional;

/** Coordinates temporary native graph copies across concurrent segment flushes. */
final class GraphCopyMemoryBudget {
  private static final GraphCopyMemoryBudget SHARED = new GraphCopyMemoryBudget();

  private long reservedCopyBytes;
  private long activeBudgetBytes = -1;

  static GraphCopyMemoryBudget shared() {
    return SHARED;
  }

  /**
   * Tries to reserve the raw INT32 payload of one temporary device-to-host adjacency copy.
   * Reservations are shared by callers in this class loader. Overlapping reservations must use
   * the same configured ceiling; a caller with a different ceiling is denied until the active
   * reservations are released. This keeps one caller from silently raising another caller's
   * active aggregate ceiling.
   */
  synchronized Optional<Reservation> tryReserve(
      long rows, long columns, long configuredBudgetBytes) {
    long requiredCopyBytes = requiredCopyBytes(rows, columns);
    if (requiredCopyBytes < 0 || configuredBudgetBytes < 0) {
      return Optional.empty();
    }
    if (requiredCopyBytes > configuredBudgetBytes) {
      return Optional.empty();
    }
    if (reservedCopyBytes != 0 && activeBudgetBytes != configuredBudgetBytes) {
      return Optional.empty();
    }
    if (reservedCopyBytes > configuredBudgetBytes
        || requiredCopyBytes > configuredBudgetBytes - reservedCopyBytes) {
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
      activeBudgetBytes = -1;
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

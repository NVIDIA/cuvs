/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.sun.management.OperatingSystemMXBean;
import java.lang.management.ManagementFactory;
import java.util.Objects;
import java.util.Optional;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.hnsw.NeighborArray;

/** Coordinates temporary native graph copies across concurrent segment flushes. */
final class GraphCopyMemoryBudget {
  private static final long NEIGHBOR_ARRAY_SHALLOW_BYTES =
      RamUsageEstimator.shallowSizeOfInstance(NeighborArray.class);

  private static final GraphCopyMemoryBudget SYSTEM =
      new GraphCopyMemoryBudget(GraphCopyMemoryBudget::readSystemMemory);

  private final MemoryProbe memoryProbe;
  private long reservedHeadroomBytes;

  GraphCopyMemoryBudget(MemoryProbe memoryProbe) {
    this.memoryProbe = Objects.requireNonNull(memoryProbe);
  }

  static GraphCopyMemoryBudget system() {
    return SYSTEM;
  }

  /**
   * Tries to reserve enough observed free memory for one graph copy and its materialized graph.
   * Callers in the same class loader share reservations. This is cooperative admission control,
   * not an operating-system memory guarantee.
   */
  synchronized Optional<Reservation> tryReserve(long rows, long columns) {
    long requiredHeadroom = requiredHeadroom(rows, columns);
    if (requiredHeadroom < 0) {
      return Optional.empty();
    }

    MemorySnapshot memory;
    try {
      memory = memoryProbe.read();
    } catch (RuntimeException unavailable) {
      return Optional.empty();
    }
    if (memory == null
        || memory.totalBytes() <= 0
        || memory.freeBytes() < 0
        || memory.freeBytes() > memory.totalBytes()) {
      return Optional.empty();
    }

    if (reservedHeadroomBytes > memory.freeBytes()
        || requiredHeadroom > memory.freeBytes() - reservedHeadroomBytes) {
      return Optional.empty();
    }
    reservedHeadroomBytes += requiredHeadroom;
    return Optional.of(new Reservation(this, requiredHeadroom));
  }

  /**
   * Estimates peak allocation from the actual matrix shape and current JVM object layout. Besides
   * the native host copy and materialized Lucene graph, one adjacency-sized allowance protects
   * against allocation races and estimation error while the copy is in flight.
   */
  static long requiredHeadroom(long rows, long columns) {
    if (rows <= 0 || rows > Integer.MAX_VALUE || columns <= 0 || columns > Integer.MAX_VALUE) {
      return -1;
    }
    try {
      long adjacencyBytes = Math.multiplyExact(Math.multiplyExact(rows, columns), Integer.BYTES);
      long neighborReferences = arraySize(rows, RamUsageEstimator.NUM_BYTES_OBJECT_REF);
      long nodeIds = arraySize(columns, Integer.BYTES);
      long scores = arraySize(columns, Float.BYTES);
      long bytesPerNode = Math.addExact(NEIGHBOR_ARRAY_SHALLOW_BYTES, nodeIds);
      bytesPerNode = Math.addExact(bytesPerNode, scores);
      long luceneGraphBytes =
          Math.addExact(neighborReferences, Math.multiplyExact(rows, bytesPerNode));
      return Math.addExact(Math.multiplyExact(adjacencyBytes, 2), luceneGraphBytes);
    } catch (ArithmeticException overflow) {
      return -1;
    }
  }

  private static long arraySize(long length, int bytesPerElement) {
    long unaligned =
        Math.addExact(
            RamUsageEstimator.NUM_BYTES_ARRAY_HEADER, Math.multiplyExact(length, bytesPerElement));
    long alignment = RamUsageEstimator.NUM_BYTES_OBJECT_ALIGNMENT;
    long remainder = unaligned % alignment;
    return remainder == 0 ? unaligned : Math.addExact(unaligned, alignment - remainder);
  }

  private synchronized void release(Reservation reservation) {
    if (reservation.released) {
      return;
    }
    reservedHeadroomBytes -= reservation.headroomBytes;
    reservation.released = true;
  }

  private static MemorySnapshot readSystemMemory() {
    java.lang.management.OperatingSystemMXBean platformBean =
        ManagementFactory.getOperatingSystemMXBean();
    if (platformBean instanceof OperatingSystemMXBean osBean) {
      return new MemorySnapshot(osBean.getTotalMemorySize(), osBean.getFreeMemorySize());
    }
    return null;
  }

  @FunctionalInterface
  interface MemoryProbe {
    MemorySnapshot read();
  }

  record MemorySnapshot(long totalBytes, long freeBytes) {}

  static final class Reservation implements AutoCloseable {
    private final GraphCopyMemoryBudget budget;
    private final long headroomBytes;
    private boolean released;

    private Reservation(GraphCopyMemoryBudget budget, long headroomBytes) {
      this.budget = budget;
      this.headroomBytes = headroomBytes;
    }

    @Override
    public void close() {
      budget.release(this);
    }
  }
}

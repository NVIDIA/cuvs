/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CuVSDeviceMatrix;
import com.nvidia.cuvs.CuVSHostMatrix;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.RowView;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/** Minimal in-memory INT matrix used by graph materialization and serialization tests. */
class IntGraphTestMatrix implements CuVSMatrix {
  private final int[][] rows;

  IntGraphTestMatrix(int[][] rows) {
    this.rows = rows;
  }

  static IntGraphTestMatrix random(int rowCount, int columnCount, long seed) {
    return new IntGraphTestMatrix(randomRows(rowCount, columnCount, seed));
  }

  static int[][] randomRows(int rowCount, int columnCount, long seed) {
    Random random = new Random(seed);
    int[][] rows = new int[rowCount][columnCount];
    for (int[] row : rows) {
      for (int column = 0; column < row.length; column++) {
        row[column] = random.nextInt(rowCount);
      }
    }
    return rows;
  }

  @Override
  public long size() {
    return rows.length;
  }

  @Override
  public long columns() {
    return rows.length == 0 ? 0 : rows[0].length;
  }

  @Override
  public DataType dataType() {
    return DataType.INT;
  }

  @Override
  public RowView getRow(long row) {
    return new IntRow(rows[Math.toIntExact(row)]);
  }

  @Override
  public void toArray(int[][] target) {
    for (int row = 0; row < rows.length; row++) {
      System.arraycopy(rows[row], 0, target[row], 0, rows[row].length);
    }
  }

  @Override
  public void toArray(float[][] target) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void toArray(byte[][] target) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void toHost(CuVSHostMatrix target) {
    throw new UnsupportedOperationException();
  }

  @Override
  public CuVSHostMatrix toHost() {
    throw new UnsupportedOperationException();
  }

  @Override
  public void toDevice(CuVSDeviceMatrix target, CuVSResources resources) {
    throw new UnsupportedOperationException();
  }

  @Override
  public CuVSDeviceMatrix toDevice(CuVSResources resources) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void close() {}

  /** Reports a synthetic device shape and fails if an unapproved host copy is attempted. */
  static class DeviceMatrix extends IntGraphTestMatrix implements CuVSDeviceMatrix {
    private final long reportedColumns;

    DeviceMatrix(int[][] rows, long reportedColumns) {
      super(rows);
      this.reportedColumns = reportedColumns;
    }

    @Override
    public long columns() {
      return reportedColumns;
    }

    @Override
    public void toHost(CuVSHostMatrix target) {
      throw new AssertionError("rejected device adjacency must not be copied to host");
    }

    @Override
    public CuVSHostMatrix toHost() {
      throw new AssertionError("rejected device adjacency must not be copied to host");
    }
  }

  static class TrackingHostMatrix extends IntGraphTestMatrix implements CuVSHostMatrix {
    private final AtomicInteger closeCount;
    private final RuntimeException closeFailure;
    private final ParallelExecutionProbe executionProbe;

    TrackingHostMatrix(AtomicInteger closeCount, RuntimeException closeFailure) {
      this(new int[][] {{0}}, closeCount, closeFailure, null);
    }

    TrackingHostMatrix(
        int[][] rows,
        AtomicInteger closeCount,
        RuntimeException closeFailure,
        ParallelExecutionProbe executionProbe) {
      super(rows);
      this.closeCount = closeCount;
      this.closeFailure = closeFailure;
      this.executionProbe = executionProbe;
    }

    @Override
    public int get(int row, int column) {
      return getRow(row).getAsInt(column);
    }

    @Override
    public RowView getRow(long row) {
      if (executionProbe != null) {
        executionProbe.recordExecution();
      }
      return super.getRow(row);
    }

    @Override
    public void close() {
      closeCount.incrementAndGet();
      if (closeFailure != null) {
        throw closeFailure;
      }
    }
  }

  /** Holds the first operation until a second execution context reaches the same work. */
  static final class ParallelExecutionProbe {
    private static final long TIMEOUT_SECONDS = 10;

    private final Set<Thread> threads = ConcurrentHashMap.newKeySet();
    private final CountDownLatch parallelExecution = new CountDownLatch(1);

    void recordExecution() {
      threads.add(Thread.currentThread());
      if (threads.size() > 1) {
        parallelExecution.countDown();
      }
      try {
        if (!parallelExecution.await(TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
          throw new AssertionError("work never reached a second execution context");
        }
      } catch (InterruptedException interrupted) {
        Thread.currentThread().interrupt();
        throw new AssertionError("interrupted while observing parallel work", interrupted);
      }
    }

    int threadCount() {
      return threads.size();
    }
  }

  private static final class IntRow implements RowView {
    private final int[] values;

    private IntRow(int[] values) {
      this.values = values;
    }

    @Override
    public long size() {
      return values.length;
    }

    @Override
    public int getAsInt(long index) {
      return values[Math.toIntExact(index)];
    }

    @Override
    public float getAsFloat(long index) {
      throw new UnsupportedOperationException();
    }

    @Override
    public byte getAsByte(long index) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void toArray(int[] target) {
      System.arraycopy(values, 0, target, 0, values.length);
    }

    @Override
    public void toArray(float[] target) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void toArray(byte[] target) {
      throw new UnsupportedOperationException();
    }
  }
}

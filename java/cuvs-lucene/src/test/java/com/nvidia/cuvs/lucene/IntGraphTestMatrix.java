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

/** Small array-backed matrix used by graph conversion tests. */
class IntGraphTestMatrix implements CuVSMatrix {
  private final int[][] rows;
  private final long reportedColumns;

  IntGraphTestMatrix(int[][] rows) {
    this(rows, rows.length == 0 ? 0 : rows[0].length);
  }

  IntGraphTestMatrix(int[][] rows, long reportedColumns) {
    this.rows = rows;
    this.reportedColumns = reportedColumns;
  }

  @Override
  public long size() {
    return rows.length;
  }

  @Override
  public long columns() {
    return reportedColumns;
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

  static class Device extends IntGraphTestMatrix implements CuVSDeviceMatrix {
    Device(int[][] rows, long reportedColumns) {
      super(rows, reportedColumns);
    }

    @Override
    public void toHost(CuVSHostMatrix target) {
      throw new AssertionError("oversized device graph must not be copied to host");
    }
  }

  private record IntRow(int[] values) implements RowView {
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

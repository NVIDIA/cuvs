/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

/** Parameters for product quantizer training. */
public class ProductQuantizerParams {
  private final int pqBits;
  private final int pqDim;
  private final boolean useSubspaces;
  private final boolean useVq;
  private final int vqNCenters;
  private final int kmeansNIters;
  private final int maxTrainPointsPerPqCode;
  private final int maxTrainPointsPerVqCluster;

  private ProductQuantizerParams(Builder builder) {
    this.pqBits = builder.pqBits;
    this.pqDim = builder.pqDim;
    this.useSubspaces = builder.useSubspaces;
    this.useVq = builder.useVq;
    this.vqNCenters = builder.vqNCenters;
    this.kmeansNIters = builder.kmeansNIters;
    this.maxTrainPointsPerPqCode = builder.maxTrainPointsPerPqCode;
    this.maxTrainPointsPerVqCluster = builder.maxTrainPointsPerVqCluster;
  }

  public int getPqBits() {
    return pqBits;
  }

  public int getPqDim() {
    return pqDim;
  }

  public boolean getUseSubspaces() {
    return useSubspaces;
  }

  public boolean getUseVq() {
    return useVq;
  }

  public int getVqNCenters() {
    return vqNCenters;
  }

  public int getKmeansNIters() {
    return kmeansNIters;
  }

  public int getMaxTrainPointsPerPqCode() {
    return maxTrainPointsPerPqCode;
  }

  public int getMaxTrainPointsPerVqCluster() {
    return maxTrainPointsPerVqCluster;
  }

  public static class Builder {
    private int pqBits = 8;
    private int pqDim = 0;
    private boolean useSubspaces = true;
    private boolean useVq = false;
    private int vqNCenters = 0;
    private int kmeansNIters = 25;
    private int maxTrainPointsPerPqCode = 256;
    private int maxTrainPointsPerVqCluster = 1024;

    public Builder withPqBits(int value) {
      this.pqBits = value;
      return this;
    }

    public Builder withPqDim(int value) {
      this.pqDim = value;
      return this;
    }

    public Builder withUseSubspaces(boolean value) {
      this.useSubspaces = value;
      return this;
    }

    public Builder withUseVq(boolean value) {
      this.useVq = value;
      return this;
    }

    public Builder withVqNCenters(int value) {
      this.vqNCenters = value;
      return this;
    }

    public Builder withKmeansNIters(int value) {
      this.kmeansNIters = value;
      return this;
    }

    public Builder withMaxTrainPointsPerPqCode(int value) {
      this.maxTrainPointsPerPqCode = value;
      return this;
    }

    public Builder withMaxTrainPointsPerVqCluster(int value) {
      this.maxTrainPointsPerVqCluster = value;
      return this;
    }

    public ProductQuantizerParams build() {
      return new ProductQuantizerParams(this);
    }
  }
}

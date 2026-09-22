/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

/**
 * Parameters for ACE (Augmented Core Extraction) graph build for HNSW.
 * ACE enables building indexes for datasets too large to fit in GPU memory by:
 * 1. Partitioning the dataset in core and augmented partitions using balanced k-means
 * 2. Building sub-indexes for each partition independently
 * 3. Concatenating sub-graphs into a final unified index
 *
 * @since 25.02
 */
public class HnswAceParams {

  private long npartitions;
  private String buildDir;
  private boolean useDisk;
  private double maxHostMemoryGb;
  private double maxGpuMemoryGb;

  private HnswAceParams(
      long npartitions,
      String buildDir,
      boolean useDisk,
      double maxHostMemoryGb,
      double maxGpuMemoryGb) {
    this.npartitions = npartitions;
    this.buildDir = buildDir;
    this.useDisk = useDisk;
    this.maxHostMemoryGb = maxHostMemoryGb;
    this.maxGpuMemoryGb = maxGpuMemoryGb;
  }

  /**
   * Gets the number of partitions for ACE partitioned build.
   *
   * @return the number of partitions
   */
  public long getNpartitions() {
    return npartitions;
  }

  /**
   * Gets the directory to store ACE build artifacts.
   *
   * @return the build directory path
   */
  public String getBuildDir() {
    return buildDir;
  }

  /**
   * Gets whether disk-based storage is enabled for ACE build.
   *
   * @return true if disk mode is enabled
   */
  public boolean isUseDisk() {
    return useDisk;
  }

  /**
   * Gets the maximum host memory limit in GiB.
   *
   * @return the max host memory limit (0 means use available memory)
   */
  public double getMaxHostMemoryGb() {
    return maxHostMemoryGb;
  }

  /**
   * Gets the maximum GPU memory limit in GiB.
   *
   * @return the max GPU memory limit (0 means use available memory)
   */
  public double getMaxGpuMemoryGb() {
    return maxGpuMemoryGb;
  }

  @Override
  public String toString() {
    return "HnswAceParams [npartitions="
        + npartitions
        + ", buildDir="
        + buildDir
        + ", useDisk="
        + useDisk
        + ", maxHostMemoryGb="
        + maxHostMemoryGb
        + ", maxGpuMemoryGb="
        + maxGpuMemoryGb
        + "]";
  }

  /**
   * Builder configures and creates an instance of {@link HnswAceParams}.
   */
  public static class Builder {

    private long npartitions = 0;
    private String buildDir = "/tmp/hnsw_ace_build";
    private boolean useDisk = false;
    private double maxHostMemoryGb = 0;
    private double maxGpuMemoryGb = 0;

    /**
     * Constructs this Builder.
     */
    public Builder() {}

    /**
     * Sets the number of partitions for ACE partitioned build.
     *
     * When set to 0 (default), the number of partitions is automatically derived
     * based on available host and GPU memory to maximize partition size while
     * ensuring the build fits in memory.
     *
     * Small values might improve recall but potentially degrade performance.
     * The partition size is on average 2 * (n_rows / npartitions) * dim *
     * sizeof(T). 2 is because of the core and augmented vectors. Please account
     * for imbalance in the partition sizes (up to 3x in our tests).
     *
     * If the specified number of partitions results in partitions that exceed
     * available memory, the value will be automatically increased to fit memory
     * constraints and a warning will be issued.
     *
     * @param npartitions the number of partitions
     * @return an instance of Builder
     */
    public Builder withNpartitions(long npartitions) {
      this.npartitions = npartitions;
      return this;
    }

    /**
     * Sets the directory to store ACE build artifacts.
     * Used when useDisk is true or when the graph does not fit in memory.
     *
     * @param buildDir the build directory path
     * @return an instance of Builder
     */
    public Builder withBuildDir(String buildDir) {
      this.buildDir = buildDir;
      return this;
    }

    /**
     * Sets whether to use disk-based storage for ACE build.
     * When true, enables disk-based operations for memory-efficient graph construction.
     *
     * @param useDisk true to enable disk mode
     * @return an instance of Builder
     */
    public Builder withUseDisk(boolean useDisk) {
      this.useDisk = useDisk;
      return this;
    }

    /**
     * Sets the maximum host memory to use for ACE build in GiB.
     *
     * When set to 0 (default), uses available host memory.
     * Useful for testing or when running alongside other memory-intensive processes.
     *
     * @param maxHostMemoryGb the max host memory in GiB
     * @return an instance of Builder
     */
    public Builder withMaxHostMemoryGb(double maxHostMemoryGb) {
      this.maxHostMemoryGb = maxHostMemoryGb;
      return this;
    }

    /**
     * Sets the maximum GPU memory to use for ACE build in GiB.
     *
     * When set to 0 (default), uses available GPU memory.
     * Useful for testing or when running alongside other memory-intensive processes.
     *
     * @param maxGpuMemoryGb the max GPU memory in GiB
     * @return an instance of Builder
     */
    public Builder withMaxGpuMemoryGb(double maxGpuMemoryGb) {
      this.maxGpuMemoryGb = maxGpuMemoryGb;
      return this;
    }

    /**
     * Builds an instance of {@link HnswAceParams}.
     *
     * @return an instance of {@link HnswAceParams}
     */
    public HnswAceParams build() {
      return new HnswAceParams(npartitions, buildDir, useDisk, maxHostMemoryGb, maxGpuMemoryGb);
    }
  }
}

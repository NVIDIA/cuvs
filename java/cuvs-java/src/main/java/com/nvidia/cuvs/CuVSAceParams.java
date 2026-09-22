/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

/**
 * Parameters for ACE (Augmented Core Extraction) graph build algorithm.
 * ACE enables building indexes for datasets too large to fit in GPU memory by:
 * 1. Partitioning the dataset in core (closest) and augmented (second-closest)
 * partitions using balanced k-means.
 * 2. Building sub-indexes for each partition independently
 * 3. Concatenating sub-graphs into a final unified index
 *
 * @since 25.12
 */
public class CuVSAceParams {

  /**
   * Number of partitions for ACE (Augmented Core Extraction) partitioned build.
   *
   * When set to 0 (default), the number of partitions is automatically derived
   * based on available host and GPU memory to maximize partition size while
   * ensuring the build fits in memory.
   *
   * Small values might improve recall but potentially degrade performance and increase memory usage.
   * Partitions should not be too small to prevent issues in KNN graph construction. The partition
   * size is on average {@code 2 * (n_rows / npartitions) * dim * sizeof(T)}—the factor 2 accounts
   * for core and augmented vectors. Please account for imbalance in the partition sizes (up to 3x in
   * our tests).
   *
   * If the specified number of partitions results in partitions that exceed
   * available memory, the value will be automatically increased to fit memory
   * constraints and a warning will be issued.
   */
  private final long npartitions;

  /**
   * The index quality for the ACE build.
   *
   * Bigger values increase the index quality. At some point, increasing this will no longer improve
   * the quality.
   */
  private final long efConstruction;

  /**
   * Directory to store ACE build artifacts (e.g., KNN graph, optimized graph).
   *
   * Used when {@link #isUseDisk()} is true or when the graph does not fit in host and GPU memory.
   * This should be the fastest disk in the system and hold enough space for twice the dataset, final
   * graph, and label mapping.
   */
  private final String buildDir;

  /**
   * Whether to use disk-based storage for ACE builds.
   *
   * When true, enables disk-based operations for memory-efficient graph construction.
   */
  private final boolean useDisk;

  /**
   * Maximum host memory to use for ACE build in GiB.
   *
   * When set to 0 (default), uses available host memory.
   * Useful for testing or when running alongside other memory-intensive processes.
   */
  private final double maxHostMemoryGb;

  /**
   * Maximum GPU memory to use for ACE build in GiB.
   *
   * When set to 0 (default), uses available GPU memory.
   * Useful for testing or when running alongside other memory-intensive processes.
   */
  private final double maxGpuMemoryGb;

  private CuVSAceParams(
      long npartitions,
      long efConstruction,
      String buildDir,
      boolean useDisk,
      double maxHostMemoryGb,
      double maxGpuMemoryGb) {
    this.npartitions = npartitions;
    this.efConstruction = efConstruction;
    this.buildDir = buildDir;
    this.useDisk = useDisk;
    this.maxHostMemoryGb = maxHostMemoryGb;
    this.maxGpuMemoryGb = maxGpuMemoryGb;
  }

  /**
   * Gets the number of partitions.
   *
   * @return the number of partitions
   */
  public long getNpartitions() {
    return npartitions;
  }

  /**
   * Gets the {@code ef_construction} parameter.
   *
   * @return the {@code ef_construction} parameter
   */
  public long getEfConstruction() {
    return efConstruction;
  }

  /**
   * Gets the build directory path.
   *
   * @return the build directory path
   */
  public String getBuildDir() {
    return buildDir;
  }

  /**
   * Gets whether disk-based mode is enabled.
   *
   * @return true if disk-based mode is enabled
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
    return "CuVSAceParams [npartitions="
        + npartitions
        + ", efConstruction="
        + efConstruction
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
   * Builder configures and creates an instance of {@link CuVSAceParams}.
   */
  public static class Builder {

    /** Number of partitions to split the dataset into (0 = auto-derive based on memory) */
    private long npartitions = 0;

    /** ef_construction parameter for HNSW used in ACE */
    private long efConstruction = 120;

    /** Directory to store intermediate build files */
    private String buildDir = "/tmp/ace_build";

    /** Whether to use disk-based mode for very large datasets */
    private boolean useDisk = false;

    /** Maximum host memory in GiB (0 = use available memory) */
    private double maxHostMemoryGb = 0;

    /** Maximum GPU memory in GiB (0 = use available memory) */
    private double maxGpuMemoryGb = 0;

    public Builder() {}

    /**
     * Sets the number of partitions.
     *
     * @param npartitions the number of partitions
     * @return an instance of Builder
     */
    public Builder withNpartitions(long npartitions) {
      this.npartitions = npartitions;
      return this;
    }

    /**
     * Sets the ef_construction parameter.
     *
     * @param efConstruction the ef_construction parameter
     * @return an instance of Builder
     */
    public Builder withEfConstruction(long efConstruction) {
      this.efConstruction = efConstruction;
      return this;
    }

    /**
     * Sets the build directory path.
     *
     * @param buildDir the build directory path
     * @return an instance of Builder
     */
    public Builder withBuildDir(String buildDir) {
      this.buildDir = buildDir;
      return this;
    }

    /**
     * Sets whether to use disk-based mode.
     *
     * @param useDisk whether to use disk-based mode
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
     * Builds an instance of {@link CuVSAceParams}.
     *
     * @return an instance of {@link CuVSAceParams}
     */
    public CuVSAceParams build() {
      return new CuVSAceParams(
          npartitions, efConstruction, buildDir, useDisk, maxHostMemoryGb, maxGpuMemoryGb);
    }
  }
}

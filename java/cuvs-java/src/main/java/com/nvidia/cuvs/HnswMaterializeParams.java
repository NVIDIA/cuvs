/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

/**
 * Parameters for materializing a layered HNSW artifact into a standard hnswlib
 * index file on disk.
 *
 * @since 26.12
 */
public class HnswMaterializeParams {

  private final String datasetPath;
  private final double maxHostMemoryGb;
  private final int numThreads;

  private HnswMaterializeParams(String datasetPath, double maxHostMemoryGb, int numThreads) {
    this.datasetPath = datasetPath;
    this.maxHostMemoryGb = maxHostMemoryGb;
    this.numThreads = numThreads;
  }

  /**
   * Gets the local dataset path holding the original-ID-ordered vectors used to
   * build the artifact.
   *
   * @return the dataset path, or null if not set
   */
  public String getDatasetPath() {
    return datasetPath;
  }

  /**
   * Gets the upper bound on host memory (in GiB) used for the base-topology
   * reorder buffer. When {@code <= 0}, the whole base topology is reordered in a
   * single in-memory pass.
   *
   * @return the max host memory in GiB (0 means a single in-memory pass)
   */
  public double getMaxHostMemoryGb() {
    return maxHostMemoryGb;
  }

  /**
   * Gets the number of host threads to use. When 0, the maximum number of
   * threads is used.
   *
   * @return the number of threads
   */
  public int getNumThreads() {
    return numThreads;
  }

  @Override
  public String toString() {
    return "HnswMaterializeParams [datasetPath="
        + datasetPath
        + ", maxHostMemoryGb="
        + maxHostMemoryGb
        + ", numThreads="
        + numThreads
        + "]";
  }

  /**
   * Builder configures and creates an instance of {@link HnswMaterializeParams}.
   */
  public static class Builder {

    private String datasetPath;
    private double maxHostMemoryGb = 0;
    private int numThreads = 0;

    /**
     * Constructs this Builder.
     */
    public Builder() {}

    /**
     * Sets the local dataset path holding the original-ID-ordered vectors used to
     * build the artifact. Supported formats match layered deserialization:
     * {@code .npy} and ANN benchmark {@code *.bin} files with a
     * {@code [uint32 rows, uint32 cols]} header ({@code .fbin}, {@code .f16bin},
     * {@code .u8bin}, {@code .i8bin}).
     *
     * @param datasetPath the local dataset path
     * @return an instance of Builder
     */
    public Builder withDatasetPath(String datasetPath) {
      this.datasetPath = datasetPath;
      return this;
    }

    /**
     * Sets the upper bound on host memory (in GiB) used for the base-topology
     * reorder buffer.
     *
     * When {@code <= 0} (default), the whole base topology is reordered in a single
     * in-memory pass (no temporary files). When set, the base topology is reordered
     * through bucketed temporary files so that peak host memory stays close to this
     * budget.
     *
     * @param maxHostMemoryGb the max host memory in GiB
     * @return an instance of Builder
     */
    public Builder withMaxHostMemoryGb(double maxHostMemoryGb) {
      this.maxHostMemoryGb = maxHostMemoryGb;
      return this;
    }

    /**
     * Sets the number of host threads to use. When 0 (default), the maximum number
     * of threads is used.
     *
     * @param numThreads the number of threads
     * @return an instance of Builder
     */
    public Builder withNumThreads(int numThreads) {
      this.numThreads = numThreads;
      return this;
    }

    /**
     * Builds an instance of {@link HnswMaterializeParams}.
     *
     * @return an instance of {@link HnswMaterializeParams}
     */
    public HnswMaterializeParams build() {
      return new HnswMaterializeParams(datasetPath, maxHostMemoryGb, numThreads);
    }
  }
}

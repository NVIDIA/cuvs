/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

/**
 * Supplemental parameters to build HNSW index.
 *
 * @since 25.02
 */
public class HnswIndexParams {

  /**
   * Distance metric types
   */
  public enum CuvsDistanceType {
    // Native values are defined by the C cuvsDistanceType enum; derive them from
    // CagraIndexParams.CuvsDistanceType so the mapping has a single source of truth.
    L2Expanded(CagraIndexParams.CuvsDistanceType.L2Expanded.value),
    InnerProduct(CagraIndexParams.CuvsDistanceType.InnerProduct.value);

    public final int value;

    private CuvsDistanceType(int value) {
      this.value = value;
    }
  }

  /**
   * Hierarchy for HNSW index when converting from CAGRA index
   *
   * NOTE: When the value is `NONE`, the HNSW index is built as a base-layer-only
   * index.
   */
  public enum CuvsHnswHierarchy {

    /**
     * Flat hierarchy, search is base-layer only
     */
    NONE(0),

    /**
     * Full hierarchy is built using the CPU
     */
    CPU(1),

    /**
     * Full hierarchy is built using the GPU
     */
    GPU(2);

    /**
     * The value for the enum choice.
     */
    public final int value;

    private CuvsHnswHierarchy(int value) {
      this.value = value;
    }
  };

  /**
   * Alias for {@link CuvsHnswHierarchy} for convenience.
   */
  public static class HnswHierarchy {
    public static final CuvsHnswHierarchy NONE = CuvsHnswHierarchy.NONE;
    public static final CuvsHnswHierarchy CPU = CuvsHnswHierarchy.CPU;
    public static final CuvsHnswHierarchy GPU = CuvsHnswHierarchy.GPU;
  }

  private CuvsHnswHierarchy hierarchy = CuvsHnswHierarchy.GPU;
  private int efConstruction = 200;
  private int numThreads = 2;
  private int vectorDimension;
  private long m = 32;
  private CuvsDistanceType metric = CuvsDistanceType.L2Expanded;
  private HnswAceParams aceParams;

  private HnswIndexParams(
      CuvsHnswHierarchy hierarchy,
      int efConstruction,
      int numThreads,
      int vectorDimension,
      long m,
      CuvsDistanceType metric,
      HnswAceParams aceParams) {
    this.hierarchy = hierarchy;
    this.efConstruction = efConstruction;
    this.numThreads = numThreads;
    this.vectorDimension = vectorDimension;
    this.m = m;
    this.metric = metric;
    this.aceParams = aceParams;
  }

  /**
   *
   * @return
   */
  public CuvsHnswHierarchy getHierarchy() {
    return hierarchy;
  }

  /**
   *
   * @return
   */
  public int getEfConstruction() {
    return efConstruction;
  }

  /**
   *
   * @return
   */
  public int getNumThreads() {
    return numThreads;
  }

  /**
   *
   * @return
   */
  public int getVectorDimension() {
    return vectorDimension;
  }

  /**
   * Gets the HNSW M parameter: number of bi-directional links per node
   * used to derive the internal GPU graph degree.
   *
   * @return the M parameter
   */
  public long getM() {
    return m;
  }

  /**
   * Gets the distance metric type.
   *
   * @return the metric type
   */
  public CuvsDistanceType getMetric() {
    return metric;
  }

  /**
   * Gets optional ACE parameters for partitioned or disk-backed GPU graph building.
   *
   * @return the ACE parameters, or null if not set
   */
  public HnswAceParams getAceParams() {
    return aceParams;
  }

  @Override
  public String toString() {
    return "HnswIndexParams [hierarchy="
        + hierarchy
        + ", efConstruction="
        + efConstruction
        + ", numThreads="
        + numThreads
        + ", vectorDimension="
        + vectorDimension
        + ", m="
        + m
        + ", metric="
        + metric
        + ", aceParams="
        + aceParams
        + "]";
  }

  /**
   * Builder configures and creates an instance of {@link HnswIndexParams}.
   */
  public static class Builder {

    private CuvsHnswHierarchy hierarchy = CuvsHnswHierarchy.GPU;
    private int efConstruction = 200;
    private int numThreads = 2;
    private int vectorDimension;
    private long m = 32;
    private CuvsDistanceType metric = CuvsDistanceType.L2Expanded;
    private HnswAceParams aceParams;

    /**
     * Constructs this Builder with an instance of Arena.
     */
    public Builder() {}

    /**
     * Sets the hierarchy for HNSW index when converting from CAGRA index.
     *
     * NOTE: When the value is `NONE`, the HNSW index is built as a base-layer-only
     * index.
     *
     * @param hierarchy the hierarchy for HNSW index when converting from CAGRA
     *                  index
     * @return an instance of Builder
     */
    public Builder withHierarchy(CuvsHnswHierarchy hierarchy) {
      this.hierarchy = hierarchy;
      return this;
    }

    /**
     * Sets the size of the candidate list during index construction.
     *
     * @param efConstruction the size of the candidate list during index construction
     * @return an instance of Builder
     */
    public Builder withEfConstruction(int efConstruction) {
      this.efConstruction = efConstruction;
      return this;
    }

    /**
     * Sets the number of host threads used during construction when hierarchy is
     * `CPU` or `GPU`. When the value is 0, the number of threads is automatically
     * determined to the maximum number of threads available. The default is 2.
     *
     * @param numThreads the number of threads
     * @return an instance of Builder
     */
    public Builder withNumThreads(int numThreads) {
      this.numThreads = numThreads;
      return this;
    }

    /**
     * Sets the vector dimension
     *
     * @param vectorDimension the vector dimension
     * @return an instance of Builder
     */
    public Builder withVectorDimension(int vectorDimension) {
      this.vectorDimension = vectorDimension;
      return this;
    }

    /**
     * Sets the HNSW M parameter: number of bi-directional links per node
     * used to derive the internal GPU graph degree.
     *
     * @param m the M parameter
     * @return an instance of Builder
     */
    public Builder withM(long m) {
      this.m = m;
      return this;
    }

    /**
     * Sets the distance metric type.
     *
     * @param metric the metric type
     * @return an instance of Builder
     */
    public Builder withMetric(CuvsDistanceType metric) {
      this.metric = metric;
      return this;
    }

    /**
     * Sets optional ACE parameters for partitioned or disk-backed GPU graph building.
     *
     * @param aceParams the ACE parameters
     * @return an instance of Builder
     */
    public Builder withAceParams(HnswAceParams aceParams) {
      this.aceParams = aceParams;
      return this;
    }

    /**
     * Builds an instance of {@link HnswIndexParams}.
     *
     * @return an instance of {@link HnswIndexParams}
     */
    public HnswIndexParams build() {
      return new HnswIndexParams(
          hierarchy,
          efConstruction,
          numThreads,
          vectorDimension,
          m,
          metric,
          aceParams);
    }
  }
}

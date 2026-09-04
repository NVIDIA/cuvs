/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import java.util.Arrays;
import java.util.Objects;

/**
 * Supplemental parameters to build a Vamana index.
 * <p>
 * The defaults match the native {@code cuvs::neighbors::vamana::index_params}
 * defaults.
 *
 * @since 26.10
 */
public class VamanaIndexParams {

  /**
   * The graph degrees the native builder supports, matching {@code DEGREE_SIZES}
   * in the cuVS Vamana implementation. Sorted, so it can be searched.
   */
  private static final int[] SUPPORTED_GRAPH_DEGREES = {32, 64, 128, 256};

  /**
   * Returns the graph degrees the native Vamana builder supports.
   *
   * @return a copy of the supported graph degrees, in ascending order
   */
  public static int[] supportedGraphDegrees() {
    return SUPPORTED_GRAPH_DEGREES.clone();
  }

  /**
   * Distance metric types supported by the Vamana builder.
   * <p>
   * The native build kernel accepts these two only. Other metrics fail inside
   * the kernel rather than at parameter validation time, so they are not
   * exposed here.
   */
  public enum CuvsDistanceType {
    /**
     * Squared L2.
     */
    L2Expanded(0),

    /**
     * Euclidean, the square root of {@link #L2Expanded}.
     */
    L2SqrtExpanded(1);

    /**
     * The value for the enum choice.
     */
    public final int value;

    private CuvsDistanceType(int value) {
      this.value = value;
    }
  }

  private final int graphDegree;
  private final int visitedSize;
  private final float vamanaIters;
  private final float alpha;
  private final float maxFraction;
  private final float batchBase;
  private final int queueSize;
  private final int reverseBatchSize;
  private final CuvsDistanceType metric;

  private VamanaIndexParams(
      int graphDegree,
      int visitedSize,
      float vamanaIters,
      float alpha,
      float maxFraction,
      float batchBase,
      int queueSize,
      int reverseBatchSize,
      CuvsDistanceType metric) {
    this.graphDegree = graphDegree;
    this.visitedSize = visitedSize;
    this.vamanaIters = vamanaIters;
    this.alpha = alpha;
    this.maxFraction = maxFraction;
    this.batchBase = batchBase;
    this.queueSize = queueSize;
    this.reverseBatchSize = reverseBatchSize;
    this.metric = metric;
  }

  /**
   * Gets the maximum degree of the output graph, the R parameter in the Vamana
   * literature.
   */
  public int getGraphDegree() {
    return graphDegree;
  }

  /**
   * Gets the maximum number of visited nodes per search, the L parameter in the
   * Vamana literature.
   */
  public int getVisitedSize() {
    return visitedSize;
  }

  /**
   * Gets the number of Vamana vector insertion iterations.
   */
  public float getVamanaIters() {
    return vamanaIters;
  }

  /**
   * Gets the alpha pruning parameter.
   */
  public float getAlpha() {
    return alpha;
  }

  /**
   * Gets the maximum fraction of the dataset inserted per batch.
   */
  public float getMaxFraction() {
    return maxFraction;
  }

  /**
   * Gets the growth rate base for batch sizes.
   */
  public float getBatchBase() {
    return batchBase;
  }

  /**
   * Gets the candidate queue size.
   */
  public int getQueueSize() {
    return queueSize;
  }

  /**
   * Gets the maximum batch size of reverse edge processing.
   */
  public int getReverseBatchSize() {
    return reverseBatchSize;
  }

  /**
   * Gets the distance metric.
   */
  public CuvsDistanceType getMetric() {
    return metric;
  }

  @Override
  public String toString() {
    return "VamanaIndexParams [graphDegree="
        + graphDegree
        + ", visitedSize="
        + visitedSize
        + ", vamanaIters="
        + vamanaIters
        + ", alpha="
        + alpha
        + ", maxFraction="
        + maxFraction
        + ", batchBase="
        + batchBase
        + ", queueSize="
        + queueSize
        + ", reverseBatchSize="
        + reverseBatchSize
        + ", metric="
        + metric
        + "]";
  }

  /**
   * Builder configures and creates an instance of {@link VamanaIndexParams}.
   */
  public static class Builder {

    private int graphDegree = 32;
    private int visitedSize = 64;
    private float vamanaIters = 1.0f;
    private float alpha = 1.2f;
    private float maxFraction = 0.06f;
    private float batchBase = 2.0f;
    private int queueSize = 127;
    private int reverseBatchSize = 1000000;
    private CuvsDistanceType metric = CuvsDistanceType.L2Expanded;

    public Builder() {}

    /**
     * Sets the maximum degree of the output graph.
     *
     * @param graphDegree the graph degree, one of
     *                    {@link VamanaIndexParams#supportedGraphDegrees()}
     * @return an instance of this Builder
     */
    public Builder withGraphDegree(int graphDegree) {
      this.graphDegree = graphDegree;
      return this;
    }

    /**
     * Sets the maximum number of visited nodes per search.
     * <p>
     * The native builder requires this to be greater than the graph degree.
     *
     * @param visitedSize the visited size
     * @return an instance of this Builder
     */
    public Builder withVisitedSize(int visitedSize) {
      this.visitedSize = visitedSize;
      return this;
    }

    /**
     * Sets the number of Vamana vector insertion iterations.
     *
     * @param vamanaIters the iteration count
     * @return an instance of this Builder
     */
    public Builder withVamanaIters(float vamanaIters) {
      this.vamanaIters = vamanaIters;
      return this;
    }

    /**
     * Sets the alpha pruning parameter.
     *
     * @param alpha the alpha value
     * @return an instance of this Builder
     */
    public Builder withAlpha(float alpha) {
      this.alpha = alpha;
      return this;
    }

    /**
     * Sets the maximum fraction of the dataset inserted per batch. A larger
     * batch decreases graph quality but improves build speed.
     *
     * @param maxFraction the maximum fraction
     * @return an instance of this Builder
     */
    public Builder withMaxFraction(float maxFraction) {
      this.maxFraction = maxFraction;
      return this;
    }

    /**
     * Sets the growth rate base for batch sizes.
     *
     * @param batchBase the batch base
     * @return an instance of this Builder
     */
    public Builder withBatchBase(float batchBase) {
      this.batchBase = batchBase;
      return this;
    }

    /**
     * Sets the candidate queue size. The native builder expects a value of the
     * form {@code (2^x) - 1}.
     *
     * @param queueSize the queue size
     * @return an instance of this Builder
     */
    public Builder withQueueSize(int queueSize) {
      this.queueSize = queueSize;
      return this;
    }

    /**
     * Sets the maximum batch size of reverse edge processing, which bounds the
     * memory footprint of that stage.
     *
     * @param reverseBatchSize the reverse batch size
     * @return an instance of this Builder
     */
    public Builder withReverseBatchSize(int reverseBatchSize) {
      this.reverseBatchSize = reverseBatchSize;
      return this;
    }

    /**
     * Sets the distance metric.
     *
     * @param metric the distance metric
     * @return an instance of this Builder
     */
    public Builder withMetric(CuvsDistanceType metric) {
      this.metric = metric;
      return this;
    }

    /**
     * Builds an instance of {@link VamanaIndexParams}.
     *
     * @return an instance of {@link VamanaIndexParams}
     */
    public VamanaIndexParams build() {
      validate();
      return new VamanaIndexParams(
          graphDegree,
          visitedSize,
          vamanaIters,
          alpha,
          maxFraction,
          batchBase,
          queueSize,
          reverseBatchSize,
          metric);
    }

    /**
     * Mirrors the checks the native builder performs, so that an invalid
     * configuration fails here with a readable message rather than inside a
     * GPU kernel.
     */
    private void validate() {
      if (Arrays.binarySearch(SUPPORTED_GRAPH_DEGREES, graphDegree) < 0) {
        throw new IllegalArgumentException(
            "graphDegree must be one of "
                + Arrays.toString(SUPPORTED_GRAPH_DEGREES)
                + ", was "
                + graphDegree);
      }
      if (visitedSize <= graphDegree) {
        throw new IllegalArgumentException(
            "visitedSize must be greater than graphDegree, was "
                + visitedSize
                + " with graphDegree "
                + graphDegree);
      }
      if (!Float.isFinite(vamanaIters) || vamanaIters < 1.0f) {
        throw new IllegalArgumentException(
            "vamanaIters must be finite and at least 1.0, was " + vamanaIters);
      }
      if (!Float.isFinite(alpha) || alpha <= 0.0f) {
        throw new IllegalArgumentException("alpha must be finite and positive, was " + alpha);
      }
      if (!Float.isFinite(maxFraction) || maxFraction <= 0.0f || maxFraction > 1.0f) {
        throw new IllegalArgumentException(
            "maxFraction must be finite and in (0, 1], was " + maxFraction);
      }
      if (!Float.isFinite(batchBase) || batchBase <= 1.0f) {
        throw new IllegalArgumentException(
            "batchBase must be finite and greater than 1.0, was " + batchBase);
      }
      if (queueSize <= 0 || Integer.bitCount(queueSize + 1) != 1) {
        throw new IllegalArgumentException(
            "queueSize must be positive and of the form (2^x) - 1, was " + queueSize);
      }
      if (reverseBatchSize <= 0) {
        throw new IllegalArgumentException(
            "reverseBatchSize must be positive, was " + reverseBatchSize);
      }
      Objects.requireNonNull(metric, "metric must not be null");
    }
  }
}

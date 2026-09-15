/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

/**
 * Supplemental compression parameters for CAGRA-Q PQ training via
 * {@link CagraIndex#makePqDataset(CagraIndex.PaddedDataset, CagraCompressionParams)}.
 *
 * @since 25.02
 */
public class CagraCompressionParams {

  private final int pqBits;
  private final int pqDim;
  private final int vqNCenters;
  private final int kmeansNIters;
  private final double vqKmeansTrainsetFraction;
  private final double pqKmeansTrainsetFraction;

  /**
   * Constructs an instance of CagraCompressionParams with passed search
   * parameters.
   *
   * @param pqBits                   the bit length of the vector element after
   *                                 compression by PQ
   * @param pqDim                    the dimensionality of the vector after
   *                                 compression by PQ
   * @param vqNCenters               the vector quantization (VQ) codebook size -
   *                                 number of “coarse cluster centers”
   * @param kmeansNIters             the number of iterations searching for kmeans
   *                                 centers (both VQ and PQ phases)
   * @param vqKmeansTrainsetFraction the fraction of data to use during iterative
   *                                 kmeans building (VQ phase)
   * @param pqKmeansTrainsetFraction the fraction of data to use during iterative
   *                                 kmeans building (PQ phase)
   */
  private CagraCompressionParams(
      int pqBits,
      int pqDim,
      int vqNCenters,
      int kmeansNIters,
      double vqKmeansTrainsetFraction,
      double pqKmeansTrainsetFraction) {
    this.pqBits = pqBits;
    this.pqDim = pqDim;
    this.vqNCenters = vqNCenters;
    this.kmeansNIters = kmeansNIters;
    this.vqKmeansTrainsetFraction = vqKmeansTrainsetFraction;
    this.pqKmeansTrainsetFraction = pqKmeansTrainsetFraction;
  }

  /**
   * Gets the bit length of the vector element after compression by PQ.
   *
   * @return the bit length of the vector element after compression by PQ.
   */
  public int getPqBits() {
    return pqBits;
  }

  /**
   * Gets the dimensionality of the vector after compression by PQ.
   *
   * @return the dimensionality of the vector after compression by PQ.
   */
  public int getPqDim() {
    return pqDim;
  }

  /**
   * Gets the vector quantization (VQ) codebook size - number of “coarse cluster
   * centers”.
   *
   * @return the vector quantization (VQ) codebook size - number of “coarse
   *         cluster centers”.
   */
  public int getVqNCenters() {
    return vqNCenters;
  }

  /**
   * Gets the number of iterations searching for kmeans centers (both VQ and PQ
   * phases).
   *
   * @return the number of iterations searching for kmeans centers (both VQ and PQ
   *         phases).
   */
  public int getKmeansNIters() {
    return kmeansNIters;
  }

  /**
   * Gets the fraction of data to use during iterative kmeans building (VQ phase).
   *
   * @return the fraction of data to use during iterative kmeans building (VQ
   *         phase).
   */
  public double getVqKmeansTrainsetFraction() {
    return vqKmeansTrainsetFraction;
  }

  /**
   * Gets the fraction of data to use during iterative kmeans building (PQ phase).
   *
   * @return the fraction of data to use during iterative kmeans building (PQ
   *         phase).
   */
  public double getPqKmeansTrainsetFraction() {
    return pqKmeansTrainsetFraction;
  }

  @Override
  public String toString() {
    return "CagraCompressionParams [pqBits="
        + pqBits
        + ", pqDim="
        + pqDim
        + ", vqNCenters="
        + vqNCenters
        + ", kmeansNIters="
        + kmeansNIters
        + ", vqKmeansTrainsetFraction="
        + vqKmeansTrainsetFraction
        + ", pqKmeansTrainsetFraction="
        + pqKmeansTrainsetFraction
        + "]";
  }

  /**
   * Builder configures and creates an instance of {@link CagraCompressionParams}.
   */
  public static class Builder {

    private int pqBits = 8;
    private int pqDim = 0;
    private int vqNCenters = 0;
    private int kmeansNIters = 25;
    private double vqKmeansTrainsetFraction = 0;
    private double pqKmeansTrainsetFraction = 0;

    public Builder() {}

    /**
     * Sets the bit length of the vector element after compression by PQ.
     *
     * Possible values: [4, 5, 6, 7, 8]. Hint: the smaller the ‘pq_bits’, the
     * smaller the index size and the better the search performance, but the lower
     * the recall.
     *
     * @param pqBits
     * @return an instance of Builder
     */
    public Builder withPqBits(int pqBits) {
      this.pqBits = pqBits;
      return this;
    }

    /**
     * Sets the dimensionality of the vector after compression by PQ.
     *
     * When zero, an optimal value is selected using a heuristic.
     *
     * @param pqDim
     * @return an instance of Builder
     */
    public Builder withPqDim(int pqDim) {
      this.pqDim = pqDim;
      return this;
    }

    /**
     * Sets the vector quantization (VQ) codebook size - number of “coarse cluster
     * centers”.
     *
     * When zero, an optimal value is selected using a heuristic.
     *
     * @param vqNCenters
     * @return an instance of Builder
     */
    public Builder withVqNCenters(int vqNCenters) {
      this.vqNCenters = vqNCenters;
      return this;
    }

    /**
     * Sets the number of iterations searching for kmeans centers (both VQ and PQ
     * phases).
     *
     * @param kmeansNIters
     * @return an instance of Builder
     */
    public Builder withKmeansNIters(int kmeansNIters) {
      this.kmeansNIters = kmeansNIters;
      return this;
    }

    /**
     * Sets the fraction of data to use during iterative kmeans building (VQ phase).
     *
     * When zero, an optimal value is selected using a heuristic.
     *
     * @param vqKmeansTrainsetFraction
     * @return an instance of Builder
     */
    public Builder withVqKmeansTrainsetFraction(double vqKmeansTrainsetFraction) {
      this.vqKmeansTrainsetFraction = vqKmeansTrainsetFraction;
      return this;
    }

    /**
     * Sets the fraction of data to use during iterative kmeans building (PQ phase).
     *
     * When zero, an optimal value is selected using a heuristic.
     *
     * @param pqKmeansTrainsetFraction
     * @return an instance of Builder
     */
    public Builder withPqKmeansTrainsetFraction(double pqKmeansTrainsetFraction) {
      this.pqKmeansTrainsetFraction = pqKmeansTrainsetFraction;
      return this;
    }

    /**
     * Builds an instance of {@link CagraCompressionParams}.
     *
     * @return an instance of {@link CagraCompressionParams}
     */
    public CagraCompressionParams build() {
      return new CagraCompressionParams(
          pqBits,
          pqDim,
          vqNCenters,
          kmeansNIters,
          vqKmeansTrainsetFraction,
          pqKmeansTrainsetFraction);
    }
  }
}

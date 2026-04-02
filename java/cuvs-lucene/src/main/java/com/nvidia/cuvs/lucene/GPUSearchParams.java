/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CuVSIvfPqParams;
import com.nvidia.cuvs.lucene.CuVS2510GPUVectorsWriter.IndexType;
import java.util.Objects;
import java.util.function.Supplier;

public class GPUSearchParams {

  /*
   * TODO: Update boundaries for all parameters when a consensus is reached.
   * Issue: https://github.com/rapidsai/cuvs-lucene/issues/99
   */
  public static final int MIN_WRITER_THREADS = 1;
  public static final int MAX_WRITER_THREADS = 512;
  public static final int MIN_INT_GRAPH_DEG = 2;
  public static final int MAX_INT_GRAPH_DEG = 512;
  public static final int MIN_GRAPH_DEG = 1;
  public static final int MAX_GRAPH_DEG = 512;

  public static final int DEFAULT_INT_GRAPH_DEGREE = 128;
  public static final int DEFAULT_GRAPH_DEGREE = 64;
  public static final CagraGraphBuildAlgo DEFAULT_CAGRA_GRAPH_BUILD_ALGO =
      CagraGraphBuildAlgo.NN_DESCENT;
  public static final IndexType DEFAULT_INDEX_TYPE = IndexType.CAGRA;
  public static final int DEFAULT_WRITER_THREADS = 1;

  public static final Supplier<CuVSIvfPqParams> DEFAULT_IVF_PQ_PARAMS =
      () -> {
        return new CuVSIvfPqParams.Builder().build();
      };

  private final int writerThreads;
  private final int intermediateGraphDegree;
  private final int graphdegree;
  private final CagraGraphBuildAlgo cagraGraphBuildAlgo;
  private final IndexType indexType;
  private final CuVSIvfPqParams cuVSIvfPqParams;

  /**
   * Constructs an instance of {@link GPUSearchParams} with specific parameter values.
   *
   * @param writerThreads Number of cuVS writer threads to use.
   * @param intermediateGraphDegree The intermediate graph degree while building the CAGRA index.
   * @param graphdegree The graph degree to use while building the CAGRA index.
   * @param cagraGraphBuildAlgo The CAGRA build algorithm to use.
   * @param indexType The type of index to build - CAGRA, BRUTEFORCE, or both.
   * @param cuVSIvfPqParams An instance of CuVSIvfPqParams containing IVF_PQ specific parameters.
   */
  private GPUSearchParams(
      int writerThreads,
      int intermediateGraphDegree,
      int graphdegree,
      CagraGraphBuildAlgo cagraGraphBuildAlgo,
      IndexType indexType,
      CuVSIvfPqParams cuVSIvfPqParams) {
    super();
    this.writerThreads = writerThreads;
    this.intermediateGraphDegree = intermediateGraphDegree;
    this.graphdegree = graphdegree;
    this.cagraGraphBuildAlgo = cagraGraphBuildAlgo;
    this.indexType = indexType;
    this.cuVSIvfPqParams = cuVSIvfPqParams;
  }

  /**
   * Get the cuVS writer threads parameter
   *
   * @return cuVS writer threads parameter
   */
  public int getWriterThreads() {
    return writerThreads;
  }

  /**
   * Get the intermediate graph degree
   *
   * @return the graph degree parameter
   */
  public int getIntermediateGraphDegree() {
    return intermediateGraphDegree;
  }

  /**
   * Get the graph degree
   *
   * @return the graph degree parameter
   */
  public int getGraphdegree() {
    return graphdegree;
  }

  /**
   * Get the CAGRA build algorithm parameter value
   *
   * @return the CAGRA build algorithm parameter value
   */
  public CagraGraphBuildAlgo getCagraGraphBuildAlgo() {
    return cagraGraphBuildAlgo;
  }

  /**
   * Get the index type parameter
   *
   * @return the index type parameter
   */
  public IndexType getIndexType() {
    return indexType;
  }

  /**
   * Get the instance of CuVSIvfPqParams
   *
   * @return an instance of CuVSIvfPqParams
   */
  public CuVSIvfPqParams getCuVSIvfPqParams() {
    return cuVSIvfPqParams;
  }

  @Override
  public String toString() {
    return "GPUSearchParams [writerThreads="
        + writerThreads
        + ", intermediateGraphDegree="
        + intermediateGraphDegree
        + ", graphdegree="
        + graphdegree
        + ", cagraGraphBuildAlgo="
        + cagraGraphBuildAlgo
        + ", indexType="
        + indexType
        + "]";
  }

  /**
   * Builder class for creating an instance of {@link GPUSearchParams}
   */
  public static class Builder {

    private int writerThreads = DEFAULT_WRITER_THREADS;
    private int intermediateGraphDegree = DEFAULT_INT_GRAPH_DEGREE;
    private int graphdegree = DEFAULT_GRAPH_DEGREE;
    private CagraGraphBuildAlgo cagraGraphBuildAlgo = DEFAULT_CAGRA_GRAPH_BUILD_ALGO;
    private IndexType indexType = DEFAULT_INDEX_TYPE;
    private CuVSIvfPqParams cuVSIvfPqParams = null;

    /**
     * Set the number of cuVS writer threads while building the index
     * Valid range - Minimum: {@value MIN_WRITER_THREADS}, Maximum: {@value MAX_WRITER_THREADS}
     * Default value - 64
     *
     * @param writerThreads the number of cuVS writer threads
     * @return instance of {@link Builder}
     */
    public Builder withWriterThreads(int writerThreads) {
      this.writerThreads = writerThreads;
      return this;
    }

    /**
     * Set the intermediate graph degree to use while building CAGRA index
     * Valid range - Minimum: {@value MIN_INT_GRAPH_DEG}, Maximum: {@value MAX_INT_GRAPH_DEG}
     * Default value - 128
     *
     * @param intermediateGraphDegree the intermediate graph degree parameter
     * @return instance of {@link Builder}
     */
    public Builder withIntermediateGraphDegree(int intermediateGraphDegree) {
      this.intermediateGraphDegree = intermediateGraphDegree;
      return this;
    }

    /**
     * Set the graph degree to use while building CAGRA index
     * Valid range - Minimum: {@value MIN_GRAPH_DEG}, Maximum: {@value MAX_GRAPH_DEG}
     * Default value - 64
     *
     * @param graphDegree the graph degree parameter
     * @return instance of {@link Builder}
     */
    public Builder withGraphDegree(int graphDegree) {
      this.graphdegree = graphDegree;
      return this;
    }

    /**
     * Set the CAGRA build algorithm.
     * Cannot be null, defaults to NN_DESCENT
     *
     * @param cagraGraphBuildAlgo the CAGRA build algorithm to use
     * @return instance of {@link Builder}
     */
    public Builder withCagraGraphBuildAlgo(CagraGraphBuildAlgo cagraGraphBuildAlgo) {
      this.cagraGraphBuildAlgo = cagraGraphBuildAlgo;
      return this;
    }

    /**
     * Set the type of index to build - CAGRA, BRUTEFORCE, or both.
     * Cannot be null, defaults to CAGRA
     *
     * @param indexType the type of index to build
     * @return instance of {@link Builder}
     */
    public Builder withIndexType(IndexType indexType) {
      this.indexType = indexType;
      return this;
    }

    /**
     * Set the instance of {@link CuVSIvfPqParams}
     *
     * @param cuVSIvfPqParams
     * @return instance of {@link Builder}
     */
    public Builder withCuVSIvfPqParams(CuVSIvfPqParams cuVSIvfPqParams) {
      this.cuVSIvfPqParams = cuVSIvfPqParams;
      return this;
    }

    /**
     * Validates the input parameters.
     *
     * @throws IllegalArgumentException
     */
    private void validate() throws IllegalArgumentException {
      if (writerThreads < MIN_WRITER_THREADS || writerThreads > MAX_WRITER_THREADS) {
        throw new IllegalArgumentException(
            "writerThreads not in valid range. Valid range: ["
                + MIN_WRITER_THREADS
                + ", "
                + MAX_WRITER_THREADS
                + "]");
      }
      if (intermediateGraphDegree < MIN_INT_GRAPH_DEG
          || intermediateGraphDegree > MAX_INT_GRAPH_DEG) {
        throw new IllegalArgumentException(
            "intermediateGraphDegree not in valid range. Valid range: ["
                + MIN_INT_GRAPH_DEG
                + ", "
                + MAX_INT_GRAPH_DEG
                + "]");
      }
      if (graphdegree < MIN_GRAPH_DEG || graphdegree > MAX_GRAPH_DEG) {
        throw new IllegalArgumentException(
            "graphdegree not in valid range. Valid range: ["
                + MIN_GRAPH_DEG
                + ", "
                + MAX_GRAPH_DEG
                + "]");
      }
      if (Objects.isNull(cagraGraphBuildAlgo)) {
        throw new IllegalArgumentException("cagraGraphBuildAlgo cannot be null.");
      }
      if (Objects.isNull(indexType)) {
        throw new IllegalArgumentException("indexType cannot be null.");
      }
    }

    /**
     * Creates and returns an instance of {@link GPUSearchParams}
     *
     * @return instance of {@link GPUSearchParams}
     */
    public GPUSearchParams build() {
      if (Objects.isNull(cuVSIvfPqParams)) {
        cuVSIvfPqParams = DEFAULT_IVF_PQ_PARAMS.get();
      }
      validate();
      return new GPUSearchParams(
          writerThreads,
          intermediateGraphDegree,
          graphdegree,
          cagraGraphBuildAlgo,
          indexType,
          cuVSIvfPqParams);
    }
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import com.nvidia.cuvs.CagraIndexParams.HnswHeuristicType;
import com.nvidia.cuvs.CuVSIvfPqParams;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Supplier;

public class AcceleratedHNSWParams {

  public static enum Strategy {
    /*
     * This strategy delegates the derivation of the CAGRA build parameters (graph degrees, build
     * algorithm and its parameters) to cuVS, based on HNSW-equivalent maxConn and beamWidth.
     *
     * This is the default and the recommended strategy.
     */
    HEURISTIC,
    /*
     * This is an option when the end-user would want to use custom parameter values.
     * This strategy should only be used under expert guidance.
     */
    CUSTOM
  }

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
  public static final int MIN_HNSW_LAYERS = 1;
  public static final int MAX_HNSW_LAYERS = 99;
  public static final int MIN_MAX_CONN = 1;
  public static final int MAX_MAX_CONN = 512;
  public static final int MIN_BEAM_WIDTH = 1;
  public static final int MAX_BEAM_WIDTH = 512;
  public static final int MIN_NUM_MERGE_WORKERS = 1;
  public static final int MAX_NUM_MERGE_WORKERS = 512;
  public static final int MIN_NN_DESCENT_NUM_ITERATIONS = 1;
  public static final int MAX_NN_DESCENT_NUM_ITERATIONS = 100;

  public static final int DEFAULT_WRITER_THREADS = 1;
  public static final int DEFAULT_INT_GRAPH_DEGREE = 128;
  public static final int DEFAULT_GRAPH_DEGREE = 64;
  public static final int DEFAULT_HNSW_LAYERS = 1;
  public static final int DEFAULT_MAX_CONN = 32;
  public static final int DEFAULT_BEAM_WIDTH = 32;
  public static final CagraGraphBuildAlgo DEFAULT_CAGRA_GRAPH_BUILD_ALGO =
      CagraGraphBuildAlgo.NN_DESCENT;
  public static final int DEFAULT_NUM_MERGE_WORKERS = 1;
  public static final Strategy DEFAULT_STRATEGY = Strategy.HEURISTIC;
  public static final CuvsDistanceType DEFAULT_CUVS_DISTANCE_TYPE = CuvsDistanceType.L2Expanded;
  public static final int DEFAULT_NN_DESCENT_NUM_ITERATIONS = 20;
  public static final HnswHeuristicType DEFAULT_HNSW_HEURISTIC_TYPE =
      HnswHeuristicType.SAME_GRAPH_FOOTPRINT;

  public static final Supplier<CuVSIvfPqParams> DEFAULT_IVF_PQ_PARAMS =
      () -> {
        return new CuVSIvfPqParams.Builder().build();
      };

  public static final Supplier<ExecutorService> DEFAULT_MERGE_EXE_SRVC =
      () -> {
        return Executors.newFixedThreadPool(DEFAULT_NUM_MERGE_WORKERS);
      };

  private final int writerThreads;
  private final int intermediateGraphDegree;
  private final int graphdegree;
  private final int hnswLayers;
  private final int maxConn;
  private final int beamWidth;
  private final CagraGraphBuildAlgo cagraGraphBuildAlgo;
  private final CuVSIvfPqParams cuVSIvfPqParams;
  private final int numMergeWorkers;
  private final ExecutorService mergeExec;
  private final Strategy strategy;
  private final CuvsDistanceType cuvsDistanceType;
  private final int nnDescentNumIterations;
  private final HnswHeuristicType hnswHeuristicType;

  /**
   * Constructs an instance of {@link AcceleratedHNSWParams} with specific parameter values.
   *
   * @param writerThreads Number of cuVS writer threads to use.
   * @param intermediateGraphDegree The intermediate graph degree while building the CAGRA index.
   * @param graphdegree The graph degree to use while building the CAGRA index.
   * @param hnswLayers The number of HNSW layers to build in the HNSW index.
   * @param maxConn The max connection parameter used when building HNSW index with the fallback mechanism.
   * @param beamWidth The beam width parameter used when building HNSW index with the fallback mechanism.
   * @param cagraGraphBuildAlgo The CAGRA graph build algorithm to use [NN_DESCENT, IVF_PQ].
   * @param cuVSIvfPqParams An instance of CuVSIvfPqParams containing IVF_PQ specific parameters.
   * @param numMergeWorkers The number of merge workers to use with the fallback mechanism.
   * @param mergeExec The instance of {@link ExecutorService} to use with the fallback mechanism.
   * @param strategy either HEURISTIC [Default] that delegates the CAGRA build parameters to cuVS (derived from the HNSW-equivalent maxConn and beamWidth) or CUSTOM that uses the parameters passed through this class.
   * @param cuvsDistanceType the cuvsDistanceType. The default option is L2Expanded.
   * @param nnDescentNumIterations the number of Iterations to run if building with NN_DESCENT.
   * @param hnswHeuristicType the heuristic cuVS applies when deriving the CAGRA build parameters from maxConn and beamWidth under the HEURISTIC strategy.
   */
  private AcceleratedHNSWParams(
      int writerThreads,
      int intermediateGraphDegree,
      int graphdegree,
      int hnswLayers,
      int maxConn,
      int beamWidth,
      CagraGraphBuildAlgo cagraGraphBuildAlgo,
      CuVSIvfPqParams cuVSIvfPqParams,
      int numMergeWorkers,
      ExecutorService mergeExec,
      Strategy strategy,
      CuvsDistanceType cuvsDistanceType,
      int nnDescentNumIterations,
      HnswHeuristicType hnswHeuristicType) {
    super();
    this.writerThreads = writerThreads;
    this.intermediateGraphDegree = intermediateGraphDegree;
    this.graphdegree = graphdegree;
    this.hnswLayers = hnswLayers;
    this.maxConn = maxConn;
    this.beamWidth = beamWidth;
    this.cagraGraphBuildAlgo = cagraGraphBuildAlgo;
    this.cuVSIvfPqParams = cuVSIvfPqParams;
    this.numMergeWorkers = numMergeWorkers;
    this.mergeExec = mergeExec;
    this.strategy = strategy;
    this.cuvsDistanceType = cuvsDistanceType;
    this.nnDescentNumIterations = nnDescentNumIterations;
    this.hnswHeuristicType = hnswHeuristicType;
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
   * Get the number of HNSW layers
   *
   * @return the number of HNSW layers
   */
  public int getHnswLayers() {
    return hnswLayers;
  }

  /**
   * Get the max connection parameter
   *
   * @return the max connection parameter
   */
  public int getMaxConn() {
    return maxConn;
  }

  /**
   * Get the beam width parameter
   *
   * @return the beam width parameter
   */
  public int getBeamWidth() {
    return beamWidth;
  }

  /**
   * Get the CAGRA graph build algorithm
   *
   * @return the CAGRA graph build algorithm
   */
  public CagraGraphBuildAlgo getCagraGraphBuildAlgo() {
    return cagraGraphBuildAlgo;
  }

  /**
   * Get the instance of {@link CuVSIvfPqParams}
   *
   * @return the instance of {@link CuVSIvfPqParams}
   */
  public CuVSIvfPqParams getCuVSIvfPqParams() {
    return cuVSIvfPqParams;
  }

  /**
   * Get the number of merge workers set to be used in the fallback mechanism
   *
   * @return the number of merge workers
   */
  public int getNumMergeWorkers() {
    return numMergeWorkers;
  }

  /**
   * Get the instance of the {@link ExecutorService} to be used in the fallback mechanism
   *
   * @return the instance of the {@link ExecutorService}
   */
  public ExecutorService getMergeExec() {
    return mergeExec;
  }

  /**
   * Get the chosen strategy:
   *
   * When HEURISTIC [Default] is chosen, the CAGRA build algorithm and its indexing parameters are automatically chosen based on the size of the data set
   * When CUSTOM is chosen, the build algorithm and its parameters (either defaults or overridden values with the use of With* methods) is used internally
   *
   * @return get the chosen {@link Strategy}
   */
  public Strategy getStrategy() {
    return strategy;
  }

  /**
   * Get the cuvs distance type
   *
   * @return the distance type
   */
  public CuvsDistanceType getCuvsDistanceType() {
    return cuvsDistanceType;
  }

  /**
   * get the number of Iterations to run if building with NN_DESCENT
   *
   * @return the number of iterations for NN_DESCENT
   */
  public int getNNDescentNumIterations() {
    return nnDescentNumIterations;
  }

  /**
   * Get the heuristic cuVS applies when deriving the CAGRA build parameters from maxConn and
   * beamWidth. Only consulted under the {@link Strategy#HEURISTIC} strategy.
   *
   * @return the {@link HnswHeuristicType} to hand to cuVS
   */
  public HnswHeuristicType getHnswHeuristicType() {
    return hnswHeuristicType;
  }

  @Override
  public String toString() {
    return "AcceleratedHNSWParams [writerThreads="
        + writerThreads
        + ", intermediateGraphDegree="
        + intermediateGraphDegree
        + ", graphdegree="
        + graphdegree
        + ", hnswLayers="
        + hnswLayers
        + ", maxConn="
        + maxConn
        + ", beamWidth="
        + beamWidth
        + ", cagraGraphBuildAlgo="
        + cagraGraphBuildAlgo
        + ", cuVSIvfPqParams="
        + cuVSIvfPqParams
        + ", numMergeWorkers="
        + numMergeWorkers
        + ", mergeExec="
        + mergeExec
        + ", strategy="
        + strategy
        + ", cuvsDistanceType="
        + cuvsDistanceType
        + ", nnDescentNumIterations="
        + nnDescentNumIterations
        + ", hnswHeuristicType="
        + hnswHeuristicType
        + "]";
  }

  /**
   * Builder class for creating an instance of {@link GPUSearchParams}
   */
  public static class Builder {

    private int writerThreads = DEFAULT_WRITER_THREADS;
    private int intermediateGraphDegree = DEFAULT_INT_GRAPH_DEGREE;
    private int graphdegree = DEFAULT_GRAPH_DEGREE;
    private int hnswLayers = DEFAULT_HNSW_LAYERS;
    private int maxConn = DEFAULT_MAX_CONN;
    private int beamWidth = DEFAULT_BEAM_WIDTH;
    private CagraGraphBuildAlgo cagraGraphBuildAlgo = DEFAULT_CAGRA_GRAPH_BUILD_ALGO;
    private int numMergeWorkers = DEFAULT_NUM_MERGE_WORKERS;
    private CuVSIvfPqParams cuVSIvfPqParams = null;
    private ExecutorService mergeExec = null;
    private Strategy strategy = DEFAULT_STRATEGY;
    private CuvsDistanceType cuvsDistanceType = DEFAULT_CUVS_DISTANCE_TYPE;
    private int nnDescentNumIterations = DEFAULT_NN_DESCENT_NUM_ITERATIONS;
    private HnswHeuristicType hnswHeuristicType = DEFAULT_HNSW_HEURISTIC_TYPE;

    /**
     * Set the number of cuVS writer threads while building the index
     * Valid range - Minimum: {@value MIN_WRITER_THREADS}, Maximum: {@value MAX_WRITER_THREADS}
     * Default value - {@value DEFAULT_WRITER_THREADS}
     *
     * @param writerThreads
     * @return instance of {@link Builder}
     */
    public Builder withWriterThreads(int writerThreads) {
      this.writerThreads = writerThreads;
      return this;
    }

    /**
     * Set the intermediate graph degree to use while building CAGRA index
     * Valid range - Minimum: {@value MIN_INT_GRAPH_DEG}, Maximum: {@value MAX_INT_GRAPH_DEG}
     * Default value - {@value DEFAULT_INT_GRAPH_DEGREE}
     *
     * @param intermediateGraphDegree
     * @return instance of {@link Builder}
     */
    public Builder withIntermediateGraphDegree(int intermediateGraphDegree) {
      this.intermediateGraphDegree = intermediateGraphDegree;
      return this;
    }

    /**
     * Set the graph degree to use while building CAGRA index
     * Valid range - Minimum: {@value MIN_GRAPH_DEG}, Maximum: {@value MAX_GRAPH_DEG}
     * Default value - {@value DEFAULT_GRAPH_DEGREE}
     *
     * @param graphDegree
     * @return instance of {@link Builder}
     */
    public Builder withGraphDegree(int graphDegree) {
      this.graphdegree = graphDegree;
      return this;
    }

    /**
     * Set the number of HNSW layers to construct while building the HNSW index
     * Valid range - Minimum: {@value MIN_HNSW_LAYERS}, Maximum: {@value MAX_HNSW_LAYERS}
     * Default value - {@value DEFAULT_HNSW_LAYERS}
     *
     * @param hnswLayers the number of HNSW layers
     * @return instance of {@link Builder}
     */
    public Builder withHNSWLayer(int hnswLayers) {
      this.hnswLayers = hnswLayers;
      return this;
    }

    /**
     * Set the max connections parameter while building HNSW index with fallback mechanism
     * Valid range - Minimum: {@value MIN_MAX_CONN}, Maximum: {@value MAX_MAX_CONN}
     * Default value - {@value DEFAULT_MAX_CONN}
     *
     * @param maxConn the max connections parameter
     * @return instance of {@link Builder}
     */
    public Builder withMaxConn(int maxConn) {
      this.maxConn = maxConn;
      return this;
    }

    /**
     * Set the beam width parameter while building HNSW index with fallback mechanism
     * Valid range - Minimum: {@value MIN_BEAM_WIDTH}, Maximum: {@value MAX_BEAM_WIDTH}
     * Default value - {@value DEFAULT_BEAM_WIDTH}
     *
     * @param beamWidth the beam width parameter
     * @return instance of {@link Builder}
     */
    public Builder withBeamWidth(int beamWidth) {
      this.beamWidth = beamWidth;
      return this;
    }

    /**
     * Set the CAGRA graph build algorithm to use
     * Default value - NN_DESCENT
     *
     * @param cagraGraphBuildAlgo
     * @return instance of {@link Builder}
     */
    public Builder withCagraGraphBuildAlgo(CagraGraphBuildAlgo cagraGraphBuildAlgo) {
      this.cagraGraphBuildAlgo = cagraGraphBuildAlgo;
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
     * Set the number of merge workers to be used with the fallback mechanism
     * Default value - {@value DEFAULT_NUM_MERGE_WORKERS}
     *
     * @param numMergeWorkers number of merge workers to set
     * @return instance of {@link Builder}
     */
    public Builder withNumMergeWorkers(int numMergeWorkers) {
      this.numMergeWorkers = numMergeWorkers;
      return this;
    }

    /**
     * Set the merge executor service to be used in the fallback mechanism
     * Default value an instance with one thread
     *
     * @param mergeExec an instance of {@link ExecutorService}
     * @return instance of {@link Builder}
     */
    public Builder withMergeExecutorService(ExecutorService mergeExec) {
      this.mergeExec = mergeExec;
      return this;
    }

    /**
     * Set the chosen strategy:
     *
     * When HEURISTIC [Default] is chosen, the CAGRA build algorithm and its indexing parameters are automatically chosen based on the size of the data set
     * When CUSTOM is chosen, the build algorithm and its parameters (either defaults or overridden values with the use of With* methods) is used internally
     *
     * Valid options - HEURISTIC, CUSTOM
     * Default value - HEURISTIC
     *
     * @param strategy, the strategy to choose
     * @return instance of {@link Builder}
     */
    public Builder withStrategy(Strategy strategy) {
      this.strategy = strategy;
      return this;
    }

    /**
     * Set the CuvsDistanceType
     *
     * @param cuvsDistanceType the CuvsDistanceType to set
     * @return instance of {@link Builder}
     */
    public Builder withCuvsDistanceType(CuvsDistanceType cuvsDistanceType) {
      this.cuvsDistanceType = cuvsDistanceType;
      return this;
    }

    /**
     * Set the number of Iterations to run if building with NN_DESCENT
     *
     * Valid range - Minimum: {@value MIN_NN_DESCENT_NUM_ITERATIONS}, Maximum: {@value MAX_NN_DESCENT_NUM_ITERATIONS}
     * Default value - {@value DEFAULT_NN_DESCENT_NUM_ITERATIONS}
     *
     * @param nnDescentNumIterations number of merge workers to set
     * @return instance of {@link Builder}
     */
    public Builder withNNDescentNumIterations(int nnDescentNumIterations) {
      this.nnDescentNumIterations = nnDescentNumIterations;
      return this;
    }

    /**
     * Set the heuristic cuVS applies when deriving the CAGRA build parameters from maxConn and
     * beamWidth. Only consulted under the {@link Strategy#HEURISTIC} strategy.
     *
     * Default value - SAME_GRAPH_FOOTPRINT, which targets a CAGRA graph of the same on-disk size as
     * the equivalent HNSW graph (graph degree = 2 * maxConn).
     *
     * @param hnswHeuristicType the {@link HnswHeuristicType} to hand to cuVS
     * @return instance of {@link Builder}
     */
    public Builder withHnswHeuristicType(HnswHeuristicType hnswHeuristicType) {
      this.hnswHeuristicType = hnswHeuristicType;
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
      if (hnswLayers < MIN_HNSW_LAYERS || hnswLayers > MAX_HNSW_LAYERS) {
        throw new IllegalArgumentException(
            "hnswLayers not in valid range. Valid range: ["
                + MIN_HNSW_LAYERS
                + ", "
                + MAX_HNSW_LAYERS
                + "]");
      }
      if (maxConn < MIN_MAX_CONN || maxConn > MAX_MAX_CONN) {
        throw new IllegalArgumentException(
            "maxConn not in valid range. Valid range: ["
                + MIN_MAX_CONN
                + ", "
                + MAX_MAX_CONN
                + "]");
      }
      if (beamWidth < MIN_BEAM_WIDTH || beamWidth > MAX_BEAM_WIDTH) {
        throw new IllegalArgumentException(
            "beamWidth not in valid range. Valid range: ["
                + MIN_BEAM_WIDTH
                + ", "
                + MAX_BEAM_WIDTH
                + "]");
      }
      if (Objects.isNull(cagraGraphBuildAlgo)) {
        throw new IllegalArgumentException("cagraGraphBuildAlgo cannot be null.");
      }
      if (numMergeWorkers < MIN_NUM_MERGE_WORKERS || numMergeWorkers > MAX_NUM_MERGE_WORKERS) {
        throw new IllegalArgumentException(
            "numMergeWorkers not in valid range. Valid range: ["
                + MIN_NUM_MERGE_WORKERS
                + ", "
                + MAX_NUM_MERGE_WORKERS
                + "]");
      }
      if (Objects.isNull(strategy)) {
        throw new IllegalArgumentException("strategy cannot be null.");
      }
      if (Objects.isNull(cuvsDistanceType)) {
        throw new IllegalArgumentException("cuvsDistanceType cannot be null.");
      }
      if (Objects.isNull(hnswHeuristicType)) {
        throw new IllegalArgumentException("hnswHeuristicType cannot be null.");
      }
      if (nnDescentNumIterations < MIN_NN_DESCENT_NUM_ITERATIONS
          || nnDescentNumIterations > MAX_NN_DESCENT_NUM_ITERATIONS) {
        throw new IllegalArgumentException(
            "nnDescentNumIterations not in valid range. Valid range: ["
                + MIN_NN_DESCENT_NUM_ITERATIONS
                + ", "
                + MAX_NN_DESCENT_NUM_ITERATIONS
                + "]");
      }
    }

    /**
     * Create an instance of {@link AcceleratedHNSWParams}
     *
     * @return instance of {@link AcceleratedHNSWParams}
     */
    public AcceleratedHNSWParams build() {
      if (Objects.isNull(cuVSIvfPqParams)) {
        cuVSIvfPqParams = DEFAULT_IVF_PQ_PARAMS.get();
      }
      if (Objects.isNull(mergeExec)) {
        mergeExec = DEFAULT_MERGE_EXE_SRVC.get();
      }
      validate();
      return new AcceleratedHNSWParams(
          writerThreads,
          intermediateGraphDegree,
          graphdegree,
          hnswLayers,
          maxConn,
          beamWidth,
          cagraGraphBuildAlgo,
          cuVSIvfPqParams,
          numMergeWorkers,
          mergeExec,
          strategy,
          cuvsDistanceType,
          nnDescentNumIterations,
          hnswHeuristicType);
    }
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import com.nvidia.cuvs.spi.CuVSProvider;
import java.util.Arrays;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Supplemental parameters to build CAGRA Index.
 *
 * @since 25.02
 */
public class CagraIndexParams {

  private final CagraGraphBuildAlgo cuvsCagraGraphBuildAlgo;
  private final CuvsDistanceType cuvsDistanceType;
  private final long intermediateGraphDegree;
  private final long graphDegree;
  private final long nnDescentNiter;
  private final int numWriterThreads;
  private final CuVSIvfPqParams cuVSIvfPqParams;
  private final CuVSAceParams cuVSAceParams;
  private final CagraCompressionParams cagraCompressionParams;

  /**
   * Enum that denotes which ANN algorithm is used to build CAGRA graph.
   */
  public enum CagraGraphBuildAlgo {
    /**
     * Select build algorithm automatically
     */
    AUTO_SELECT(0),
    /**
     * Use IVF-PQ to build all-neighbors knn graph
     */
    IVF_PQ(1),
    /**
     * Experimental, use NN-Descent to build all-neighbors knn graph
     */
    NN_DESCENT(2),
    /**
     * Experimental, use ACE (Augmented Core Extraction) to build graph for large datasets.
     * 4 to be consistent with the other interfaces.
     */
    ACE(4);

    /**
     * The value for the enum choice.
     */
    public final int value;

    private CagraGraphBuildAlgo(int value) {
      this.value = value;
    }

    private static final Map<Integer, CagraGraphBuildAlgo> VALUES =
        Arrays.stream(CagraGraphBuildAlgo.values())
            .collect(Collectors.toUnmodifiableMap(x -> x.value, Function.identity()));

    public static CagraGraphBuildAlgo of(int i) {
      return VALUES.get(i);
    }
  }

  /**
   * A strategy for selecting the graph build parameters based on similar HNSW index
   * parameters.
   */
  public enum HnswHeuristicType {
    /**
     * Create a graph that is very similar to an HNSW graph in
     * terms of the number of nodes and search performance. Since HNSW produces a variable-degree
     * graph (2M being the max graph degree) and CAGRA produces a fixed-degree graph, there's always a
     * difference in the performance of the two.
     *
     * This function attempts to produce such a graph that the QPS and recall of the two graphs being
     * searched by HNSW are close for any search parameter combination. The CAGRA-produced graph tends
     * to have a "longer tail" on the low recall side (that is being slightly faster and less
     * precise).
     */
    SIMILAR_SEARCH_PERFORMANCE(0),
    /**
     * Create a graph that has the same binary size as an HNSW graph with the given parameters
     * (graph_degree = 2 * M) while trying to match the search performance as closely as possible.
     *
     * The reference HNSW index and the corresponding from-CAGRA generated HNSW index will NOT produce
     * the same recalls and QPS for the same parameter ef. The graphs are different internally. For
     * the same ef, the from-CAGRA index likely has a slightly higher recall and slightly lower QPS.
     * However, the Recall-QPS curves should be similar (i.e. the points are just shifted along the
     * curve).
     */
    SAME_GRAPH_FOOTPRINT(1);

    /**
     * The value for the enum choice.
     */
    public final int value;

    private HnswHeuristicType(int value) {
      this.value = value;
    }

    private static final Map<Integer, HnswHeuristicType> VALUES =
        Arrays.stream(HnswHeuristicType.values())
            .collect(Collectors.toUnmodifiableMap(x -> x.value, Function.identity()));

    public static HnswHeuristicType of(int i) {
      return VALUES.get(i);
    }
  }

  /**
   * Enum that denotes how to compute distance.
   */
  public enum CuvsDistanceType {

    /**
     * evaluate as dist_ij = sum(x_ik^2) + sum(y_ij)^2 - 2*sum(x_ik * y_jk)
     */
    L2Expanded(0),
    /**
     * same as above, but inside the epilogue, perform square root operation
     */
    L2SqrtExpanded(1),
    /**
     * cosine distance
     */
    CosineExpanded(2),
    /**
     * L1 distance *
     */
    L1(3),
    /**
     * evaluate as dist_ij += (x_ik - y-jk)^2 *
     */
    L2Unexpanded(4),
    /**
     * same as above, but inside the epilogue, perform square root operation
     */
    L2SqrtUnexpanded(5),
    /**
     * basic inner product
     */
    InnerProduct(6),
    /**
     * Chebyshev (Linf) distance
     */
    Linf(7),
    /**
     * Canberra distance
     */
    Canberra(8),
    /**
     * Generalized Minkowski distance
     */
    LpUnexpanded(9),
    /**
     * Correlation distance
     */
    CorrelationExpanded(10),
    /**
     * Jaccard distance
     */
    JaccardExpanded(11),
    /**
     * Hellinger distance
     */
    HellingerExpanded(12),
    /**
     * Haversine distance
     */
    Haversine(13),
    /**
     * Bray-Curtis distance
     */
    BrayCurtis(14),
    /**
     * Jensen-Shannon distance
     */
    JensenShannon(15),
    /**
     * Hamming distance
     */
    HammingUnexpanded(16),
    /**
     * KLDivergence
     */
    KLDivergence(17),
    /**
     * RusselRao
     */
    RusselRaoExpanded(18),
    /**
     * Dice-Sorensen distance
     */
    DiceExpanded(19),
    /**
     * Precomputed (special value)
     */
    Precomputed(100);

    /**
     * The value for the enum choice.
     */
    public final int value;

    private CuvsDistanceType(int value) {
      this.value = value;
    }

    private static final Map<Integer, CuvsDistanceType> VALUES =
        Arrays.stream(CuvsDistanceType.values())
            .collect(Collectors.toUnmodifiableMap(x -> x.value, Function.identity()));

    public static CuvsDistanceType of(int value) {
      return VALUES.get(value);
    }
  }

  /**
   * Enum that denotes codebook gen options.
   */
  public enum CodebookGen {
    PER_SUBSPACE(0),

    PER_CLUSTER(1);

    /**
     * The value for the enum choice.
     */
    public final int value;

    private CodebookGen(int value) {
      this.value = value;
    }

    private static final Map<Integer, CodebookGen> VALUES =
        Arrays.stream(CodebookGen.values())
            .collect(Collectors.toUnmodifiableMap(x -> x.value, Function.identity()));

    public static CodebookGen of(int value) {
      return VALUES.get(value);
    }
  }

  /**
   * Enum that denotes cuda datatypes.
   */
  public enum CudaDataType {
    CUDA_R_16F(2),

    CUDA_C_16F(6),

    CUDA_R_16BF(14),

    CUDA_C_16BF(15),

    CUDA_R_32F(0),

    CUDA_C_32F(4),

    CUDA_R_64F(1),

    CUDA_C_64F(5),

    CUDA_R_4I(16),

    CUDA_C_4I(17),

    CUDA_R_4U(18),

    CUDA_C_4U(19),

    CUDA_R_8I(3),

    CUDA_C_8I(7),

    CUDA_R_8U(8),

    CUDA_C_8U(9),

    CUDA_R_16I(20),

    CUDA_C_16I(21),

    CUDA_R_16U(22),

    CUDA_C_16U(23),

    CUDA_R_32I(10),

    CUDA_C_32I(11),

    CUDA_R_32U(12),

    CUDA_C_32U(13),

    CUDA_R_64I(24),

    CUDA_C_64I(25),

    CUDA_R_64U(26),

    CUDA_C_64U(27),

    CUDA_R_8F_E4M3(28),

    CUDA_R_8F_E5M2(29);

    public final int value;

    private CudaDataType(int value) {
      this.value = value;
    }

    private static final Map<Integer, CudaDataType> VALUES =
        Arrays.stream(CudaDataType.values())
            .collect(Collectors.toUnmodifiableMap(x -> x.value, Function.identity()));

    public static CudaDataType of(int value) {
      return VALUES.get(value);
    }
  }

  private CagraIndexParams(
      long intermediateGraphDegree,
      long graphDegree,
      CagraGraphBuildAlgo CuvsCagraGraphBuildAlgo,
      long nnDescentNiter,
      int writerThreads,
      CuvsDistanceType cuvsDistanceType,
      CuVSIvfPqParams cuVSIvfPqParams,
      CuVSAceParams cuVSAceParams,
      CagraCompressionParams cagraCompressionParams) {
    this.intermediateGraphDegree = intermediateGraphDegree;
    this.graphDegree = graphDegree;
    this.cuvsCagraGraphBuildAlgo = CuvsCagraGraphBuildAlgo;
    this.nnDescentNiter = nnDescentNiter;
    this.numWriterThreads = writerThreads;
    this.cuvsDistanceType = cuvsDistanceType;
    this.cuVSIvfPqParams = cuVSIvfPqParams;
    this.cuVSAceParams = cuVSAceParams;
    this.cagraCompressionParams = cagraCompressionParams;
  }

  public static CagraIndexParams fromHnswParams(
      long rows,
      long dim,
      int M,
      int efConstruction,
      HnswHeuristicType heuristic,
      CuvsDistanceType metric) {
    return CuVSProvider.provider()
        .cagraIndexParamsFromHnswParams(rows, dim, M, efConstruction, heuristic, metric);
  }

  public static CagraIndexParams fromDataset(
      long rows, long dim, long graphDegree, CuvsDistanceType metric, long buildQuality) {
    return CuVSProvider.provider()
        .cagraIndexParamsFromDataset(rows, dim, graphDegree, metric, buildQuality);
  }

  /**
   * Gets the degree of input graph for pruning.
   *
   * @return the degree of input graph
   */
  public long getIntermediateGraphDegree() {
    return intermediateGraphDegree;
  }

  /**
   * Gets the degree of output graph.
   *
   * @return the degree of output graph
   */
  public long getGraphDegree() {
    return graphDegree;
  }

  /**
   * Gets the {@link CagraGraphBuildAlgo} used to build the index.
   */
  public CagraGraphBuildAlgo getCagraGraphBuildAlgo() {
    return cuvsCagraGraphBuildAlgo;
  }

  /**
   * Gets the number of iterations to run if building with
   * {@link CagraGraphBuildAlgo#NN_DESCENT}
   */
  public long getNNDescentNumIterations() {
    return nnDescentNiter;
  }

  /**
   * Gets the {@link CuvsDistanceType} used to build the index.
   */
  public CuvsDistanceType getCuvsDistanceType() {
    return cuvsDistanceType;
  }

  /**
   * Gets the number of threads used to build the index.
   */
  public int getNumWriterThreads() {
    return numWriterThreads;
  }

  /**
   * Gets the IVF_PQ parameters.
   */
  public CuVSIvfPqParams getCuVSIvfPqParams() {
    return cuVSIvfPqParams;
  }

  /**
   * Gets the ACE parameters.
   */
  public CuVSAceParams getCuVSAceParams() {
    return cuVSAceParams;
  }

  /**
   * Gets the CAGRA build algorithm.
   */
  public CagraGraphBuildAlgo getCuvsCagraGraphBuildAlgo() {
    return cuvsCagraGraphBuildAlgo;
  }

  /**
   * Gets the CAGRA compression parameters.
   */
  public CagraCompressionParams getCagraCompressionParams() {
    return cagraCompressionParams;
  }

  @Override
  public String toString() {
    return "CagraIndexParams [cuvsCagraGraphBuildAlgo="
        + cuvsCagraGraphBuildAlgo
        + ", cuvsDistanceType="
        + cuvsDistanceType
        + ", intermediateGraphDegree="
        + intermediateGraphDegree
        + ", graphDegree="
        + graphDegree
        + ", nnDescentNiter="
        + nnDescentNiter
        + ", numWriterThreads="
        + numWriterThreads
        + ", cuVSIvfPqParams="
        + cuVSIvfPqParams
        + ", cuVSAceParams="
        + cuVSAceParams
        + ", cagraCompressionParams="
        + cagraCompressionParams
        + "]";
  }

  /**
   * Builder configures and creates an instance of {@link CagraIndexParams}.
   */
  public static class Builder {

    private CagraGraphBuildAlgo cuvsCagraGraphBuildAlgo = CagraGraphBuildAlgo.NN_DESCENT;
    private CuvsDistanceType cuvsDistanceType = CuvsDistanceType.L2Expanded;
    private long intermediateGraphDegree = 128;
    private long graphDegree = 64;
    private long nnDescentNumIterations = 20;
    private int numWriterThreads = 2;
    private CuVSIvfPqParams cuVSIvfPqParams = new CuVSIvfPqParams.Builder().build();
    private CuVSAceParams cuVSAceParams = new CuVSAceParams.Builder().build();
    private CagraCompressionParams cagraCompressionParams;

    public Builder() {}

    /**
     * Sets the degree of input graph for pruning.
     *
     * @param intermediateGraphDegree degree of input graph for pruning
     * @return an instance of Builder
     */
    public Builder withIntermediateGraphDegree(long intermediateGraphDegree) {
      this.intermediateGraphDegree = intermediateGraphDegree;
      return this;
    }

    /**
     * Sets the degree of output graph.
     *
     * @param graphDegree degree of output graph
     * @return an instance to Builder
     */
    public Builder withGraphDegree(long graphDegree) {
      this.graphDegree = graphDegree;
      return this;
    }

    /**
     * Sets the CuvsCagraGraphBuildAlgo to use.
     *
     * @param cuvsCagraGraphBuildAlgo the CuvsCagraGraphBuildAlgo to use
     * @return an instance of Builder
     */
    public Builder withCagraGraphBuildAlgo(CagraGraphBuildAlgo cuvsCagraGraphBuildAlgo) {
      this.cuvsCagraGraphBuildAlgo = cuvsCagraGraphBuildAlgo;
      return this;
    }

    /**
     * Sets the metric to use.
     *
     * @param cuvsDistanceType the {@link CuvsDistanceType} to use
     * @return an instance of Builder
     */
    public Builder withMetric(CuvsDistanceType cuvsDistanceType) {
      this.cuvsDistanceType = cuvsDistanceType;
      return this;
    }

    /**
     * Sets the Number of Iterations to run if building with
     * {@link CagraGraphBuildAlgo#NN_DESCENT}.
     *
     * @param nnDescentNiter number of Iterations to run if building with
     *                       {@link CagraGraphBuildAlgo#NN_DESCENT}
     * @return an instance of Builder
     */
    public Builder withNNDescentNumIterations(long nnDescentNiter) {
      this.nnDescentNumIterations = nnDescentNiter;
      return this;
    }

    /**
     * Sets the number of writer threads to use for indexing.
     *
     * @param numWriterThreads number of writer threads to use
     * @return an instance of Builder
     */
    public Builder withNumWriterThreads(int numWriterThreads) {
      this.numWriterThreads = numWriterThreads;
      return this;
    }

    /**
     * Sets the IVF_PQ index parameters.
     *
     * @param cuVSIvfPqParams the IVF_PQ index parameters
     * @return an instance of Builder
     */
    public Builder withCuVSIvfPqParams(CuVSIvfPqParams cuVSIvfPqParams) {
      this.cuVSIvfPqParams = cuVSIvfPqParams;
      return this;
    }

    /**
     * Sets the ACE index parameters.
     *
     * @param cuVSAceParams the ACE index parameters
     * @return an instance of Builder
     */
    public Builder withCuVSAceParams(CuVSAceParams cuVSAceParams) {
      this.cuVSAceParams = cuVSAceParams;
      return this;
    }

    /**
     * Registers an instance of configured {@link CagraCompressionParams} with this
     * Builder.
     *
     * @param cagraCompressionParams An instance of CagraCompressionParams.
     * @return An instance of this Builder.
     */
    public Builder withCompressionParams(CagraCompressionParams cagraCompressionParams) {
      this.cagraCompressionParams = cagraCompressionParams;
      return this;
    }

    /**
     * Builds an instance of {@link CagraIndexParams}.
     *
     * @return an instance of {@link CagraIndexParams}
     */
    public CagraIndexParams build() {
      return new CagraIndexParams(
          intermediateGraphDegree,
          graphDegree,
          cuvsCagraGraphBuildAlgo,
          nnDescentNumIterations,
          numWriterThreads,
          cuvsDistanceType,
          cuVSIvfPqParams,
          cuVSAceParams,
          cagraCompressionParams);
    }
  }
}

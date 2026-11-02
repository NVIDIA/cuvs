/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal;

import static com.nvidia.cuvs.internal.CuVSParamsHelper.createTieredIndexParams;
import static com.nvidia.cuvs.internal.common.CloseableRMMAllocation.allocateRMMSegment;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_FLOAT;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_FLOAT_BYTE_SIZE;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_INT_BYTE_SIZE;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_LONG;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_LONG_BYTE_SIZE;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_POINTER;
import static com.nvidia.cuvs.internal.common.Util.CudaMemcpyKind.*;
import static com.nvidia.cuvs.internal.common.Util.buildMemorySegment;
import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.common.Util.checkCudaError;
import static com.nvidia.cuvs.internal.common.Util.concatenate;
import static com.nvidia.cuvs.internal.common.Util.prepareTensor;
import static com.nvidia.cuvs.internal.panama.headers_h.*;

import com.nvidia.cuvs.*;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.internal.common.CloseableHandle;
import com.nvidia.cuvs.internal.common.CloseableRMMAllocation;
import com.nvidia.cuvs.internal.common.Util;
import com.nvidia.cuvs.internal.panama.cuvsCagraIndexParams;
import com.nvidia.cuvs.internal.panama.cuvsCagraSearchParams;
import com.nvidia.cuvs.internal.panama.cuvsFilter;
import com.nvidia.cuvs.internal.panama.cuvsTieredIndexParams;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.util.BitSet;
import java.util.Objects;

/**
 * {@link TieredIndex} encapsulates a Tiered index, along with methods to interact
 * with it.
 * <p>
 * TieredIndex is a hybrid index that combines brute force search for small datasets
 * with ANN algorithms (like CAGRA) for larger datasets, providing optimal performance
 * across different data sizes.
 *
 * @since 25.02
 */
public class TieredIndexImpl implements TieredIndex {
  private final CuVSMatrix dataset;
  private final CuVSResources resources;
  private final TieredIndexParams tieredIndexParameters;
  private final IndexReference tieredIndexReference;
  private boolean destroyed;

  /**
   * Constructor for building the index using specified dataset
   */
  private TieredIndexImpl(
      TieredIndexParams indexParameters, CuVSMatrix dataset, CuVSResources resources) {
    this.tieredIndexParameters = indexParameters;
    this.dataset = dataset;
    this.resources = resources;
    this.tieredIndexReference = build();
    this.destroyed = false;
  }

  /**
   * Constructor for loading the index from an {@link InputStream}
   */
  private TieredIndexImpl(InputStream inputStream, CuVSResources resources) {
    throw new UnsupportedOperationException("Deserialization of TieredIndex is not yet supported");
  }

  private void checkNotDestroyed() {
    if (destroyed) {
      throw new IllegalStateException("destroyed");
    }
  }

  /**
   * Invokes the native destroy_tiered_index to de-allocate the Tiered index
   */
  @Override
  public void close() {
    checkNotDestroyed();
    try {
      int returnValue = cuvsTieredIndexDestroy(tieredIndexReference.getMemorySegment());
      checkCuVSError(returnValue, "cuvsTieredIndexDestroy");
      if (dataset != null) {
        dataset.close();
      }
    } finally {
      destroyed = true;
    }
  }

  /**
   * Translates C build_tiered_index function to Java
   * Invokes the native build_tiered_index function via the Panama API to build the
   * {@link TieredIndex}
   *
   * @return an instance of {@link IndexReference} that holds the pointer to the
   *         index
   */
  private IndexReference build() {
    try (var localArena = Arena.ofConfined()) {
      assert dataset != null;
      long rows = dataset.size();
      long cols = dataset.columns();

      // Get host data
      MemorySegment hostDataSeg = ((CuVSMatrixInternal) dataset).memorySegment();

      try (var resourceAccess = resources.access();
          var indexParamsHandle =
              tieredIndexParameters != null
                  ? segmentFromIndexParams(localArena, tieredIndexParameters)
                  : CloseableHandle.NULL) {

        MemorySegment indexParamsMemorySegment = indexParamsHandle.handle();
        long cuvsRes = resourceAccess.handle();

        // TieredIndex REQUIRES device memory - allocate it
        long datasetSize = C_FLOAT_BYTE_SIZE * rows * cols;
        try (var datasetDP = allocateRMMSegment(cuvsRes, datasetSize)) {
          // Copy host to device
          Util.cudaMemcpy(datasetDP.handle(), hostDataSeg, datasetSize, HOST_TO_DEVICE);

          // Create tensor from device memory
          long[] datasetShape = {rows, cols};
          MemorySegment datasetTensor =
              prepareTensor(
                  localArena, datasetDP.handle(), datasetShape, kDLFloat(), 32, kDLCUDA());

          MemorySegment index = localArena.allocate(cuvsTieredIndex_t);
          var returnValue = cuvsTieredIndexCreate(index);
          checkCuVSError(returnValue, "cuvsTieredIndexCreate");

          returnValue = cuvsStreamSync(cuvsRes);
          checkCuVSError(returnValue, "cuvsStreamSync");

          // Extract the actual index pointer that was written by Create
          MemorySegment actualIndexPtr = index.get(C_POINTER, 0);

          returnValue =
              cuvsTieredIndexBuild(
                  cuvsRes, indexParamsMemorySegment, datasetTensor, actualIndexPtr);
          checkCuVSError(returnValue, "cuvsTieredIndexBuild");

          return new IndexReference(actualIndexPtr);
        }
      }
    }
  }

  private static final BitSet[] EMPTY_PREFILTER_BITSET = new BitSet[0];

  /**
   * Translates C search_tiered_index function to Java
   * Invokes the native search_tiered_index via the Panama API for searching a
   * Tiered index.
   *
   * @param query an instance of {@link TieredIndexQuery} holding the query vectors and
   *              other parameters
   * @return an instance of {@link SearchResults} containing the results
   */
  @Override
  public SearchResults search(TieredIndexQuery query) throws Throwable {
    try (var localArena = Arena.ofConfined()) {
      checkNotDestroyed();
      int topK =
          query.getMapping() != null
              ? Math.min(query.getMapping().size(), query.getTopK())
              : query.getTopK();
      long numQueries = query.getQueryVectors().length;
      long numBlocks = (long) topK * numQueries;
      int vectorDimension = numQueries > 0 ? query.getQueryVectors()[0].length : 0;

      SequenceLayout neighborsLayout = MemoryLayout.sequenceLayout(numBlocks, C_LONG);
      SequenceLayout distancesLayout = MemoryLayout.sequenceLayout(numBlocks, C_FLOAT);
      MemorySegment neighborsSeg = localArena.allocate(neighborsLayout);
      MemorySegment distancesSeg = localArena.allocate(distancesLayout);

      // Get host query data
      MemorySegment hostQueriesSeg = Util.buildMemorySegment(localArena, query.getQueryVectors());

      final long queriesBytes = C_FLOAT_BYTE_SIZE * numQueries * vectorDimension;
      final long neighborsBytes = C_LONG_BYTE_SIZE * numQueries * topK; // 64-bit for tiered index
      final long distancesBytes = C_FLOAT_BYTE_SIZE * numQueries * topK;
      final boolean hasPreFilter = query.getPrefilter() != null;
      final BitSet[] prefilters =
          hasPreFilter ? new BitSet[] {query.getPrefilter()} : EMPTY_PREFILTER_BITSET;
      final long prefilterDataLength = hasPreFilter ? query.getNumDocs() * prefilters.length : 0;
      final long prefilterLen = hasPreFilter ? (prefilterDataLength + 31) / 32 : 0;
      // TODO: is this correct? prefilters is a LONG array
      final long prefilterBytes = C_INT_BYTE_SIZE * prefilterLen;

      try (var resourceAccess = query.getResources().access()) {
        long cuvsRes = resourceAccess.handle();

        // Allocate DEVICE memory for all data
        try (var queriesDP = allocateRMMSegment(cuvsRes, queriesBytes);
            var neighborsDP = allocateRMMSegment(cuvsRes, neighborsBytes);
            var distancesDP = allocateRMMSegment(cuvsRes, distancesBytes);
            var prefilterDP =
                hasPreFilter
                    ? allocateRMMSegment(cuvsRes, prefilterBytes)
                    : CloseableRMMAllocation.EMPTY) {

          // Copy queries from host to device
          var returnValue =
              cudaMemcpy(
                  queriesDP.handle(), hostQueriesSeg, queriesBytes, cudaMemcpyHostToDevice());
          checkCudaError(returnValue, "cudaMemcpy");

          // Create tensors from device memory
          long[] queriesShape = {numQueries, vectorDimension};
          MemorySegment queriesTensor =
              prepareTensor(
                  localArena, queriesDP.handle(), queriesShape, kDLFloat(), 32, kDLCUDA());
          long[] neighborsShape = {numQueries, topK};
          MemorySegment neighborsTensor =
              prepareTensor(
                  localArena,
                  neighborsDP.handle(),
                  neighborsShape,
                  kDLInt(),
                  64,
                  kDLCUDA()); // 64-bit int
          long[] distancesShape = {numQueries, topK};
          MemorySegment distancesTensor =
              prepareTensor(
                  localArena, distancesDP.handle(), distancesShape, kDLFloat(), 32, kDLCUDA());

          // Sync before prefilter setup
          returnValue = cuvsStreamSync(cuvsRes);
          checkCuVSError(returnValue, "cuvsStreamSync");

          // Handle prefilter
          MemorySegment prefilter = cuvsFilter.allocate(localArena);

          if (hasPreFilter) {
            BitSet concatenatedFilters = concatenate(prefilters, (int) query.getNumDocs());
            long[] filters = concatenatedFilters.toLongArray();
            MemorySegment hostPrefilterSeg =
                buildMemorySegment(localArena, filters, (prefilterDataLength + 63) / 64);

            long[] prefilterShape = {prefilterLen};

            // Copy prefilter to device
            checkCudaError(
                cudaMemcpy(
                    prefilterDP.handle(),
                    hostPrefilterSeg,
                    prefilterBytes,
                    cudaMemcpyHostToDevice()),
                "cudaMemcpy");

            MemorySegment prefilterTensor =
                prepareTensor(
                    localArena, prefilterDP.handle(), prefilterShape, kDLUInt(), 32, kDLCUDA());

            cuvsFilter.type(prefilter, 1); // BITSET
            cuvsFilter.addr(prefilter, prefilterTensor.address());
          } else {
            cuvsFilter.type(prefilter, 0); // NO_FILTER
            cuvsFilter.addr(prefilter, 0);
          }

          // Perform search
          returnValue =
              cuvsTieredIndexSearch(
                  cuvsRes,
                  segmentFromSearchParams(query.getCagraSearchParameters(), localArena),
                  tieredIndexReference.getMemorySegment(),
                  queriesTensor,
                  neighborsTensor,
                  distancesTensor,
                  prefilter);
          checkCuVSError(returnValue, "cuvsTieredIndexSearch");

          // Copy results from device to host
          returnValue =
              cudaMemcpy(
                  neighborsSeg, neighborsDP.handle(), neighborsBytes, cudaMemcpyDeviceToHost());
          checkCudaError(returnValue, "cudaMemcpy");
          returnValue =
              cudaMemcpy(
                  distancesSeg, distancesDP.handle(), distancesBytes, cudaMemcpyDeviceToHost());
          checkCudaError(returnValue, "cudaMemcpy");

          return TieredSearchResultsImpl.create(
              neighborsLayout,
              distancesLayout,
              neighborsSeg,
              distancesSeg,
              topK,
              query.getMapping(),
              numQueries);
        }
      }
    }
  }

  @Override
  public ExtendBuilder extend() {
    checkNotDestroyed();
    return new ExtendBuilder(this);
  }

  /**
   * Performs the actual extend operation
   */
  private void performExtend(CuVSMatrix extendDataset) {
    try (var localArena = Arena.ofConfined()) {
      assert extendDataset != null;
      long rows = extendDataset.size();
      long cols = extendDataset.columns();

      // Get host data
      MemorySegment hostDataSeg = ((CuVSMatrixInternal) extendDataset).memorySegment();

      try (var resourceAccess = resources.access()) {
        long cuvsRes = resourceAccess.handle();

        // Allocate device memory for extend data
        long dataSize = C_FLOAT_BYTE_SIZE * rows * cols;
        try (var datasetDP = allocateRMMSegment(cuvsRes, dataSize)) {
          // Copy host to device
          checkCudaError(
              cudaMemcpy(datasetDP.handle(), hostDataSeg, dataSize, cudaMemcpyHostToDevice()),
              "cudaMemcpy");

          // Create tensor from device memory
          long[] datasetShape = {rows, cols};
          MemorySegment datasetTensor =
              prepareTensor(
                  localArena, datasetDP.handle(), datasetShape, kDLFloat(), 32, kDLCUDA());

          checkCuVSError(cuvsStreamSync(cuvsRes), "cuvsStreamSync");

          checkCuVSError(
              cuvsTieredIndexExtend(
                  cuvsRes, datasetTensor, tieredIndexReference.getMemorySegment()),
              "cuvsTieredIndexExtend");
        }
      }
    }
  }

  /**
   * ExtendBuilder implementation
   */
  public static class ExtendBuilder implements TieredIndex.ExtendBuilder {
    private final TieredIndexImpl index;
    private CuVSMatrix dataset;

    private ExtendBuilder(TieredIndexImpl index) {
      this.index = index;
    }

    @Override
    public ExtendBuilder withDataset(float[][] vectors) {
      this.dataset = CuVSMatrix.ofArray(vectors);
      return this;
    }

    @Override
    public ExtendBuilder withDataset(CuVSMatrix dataset) {
      this.dataset = dataset;
      return this;
    }

    @Override
    public void execute() {
      if (dataset == null) {
        throw new IllegalArgumentException("Must provide a dataset");
      }

      index.performExtend(dataset);
    }
  }

  /**
   * Allocates the configured index parameters in the MemorySegment.
   */

  /**
   * Allocates the configured index parameters in a MemorySegment and returns a CloseableHandle
   * for safe resource management.
   */
  private static CloseableHandle segmentFromIndexParams(Arena arena, TieredIndexParams params) {
    CloseableHandle paramsHandle = createTieredIndexParams();
    MemorySegment seg = paramsHandle.handle();

    // Get the metric from CagraParams if available, otherwise use TieredIndex metric
    int metric;
    if (params.getCagraParams() != null) {
      metric = params.getCagraParams().getCuvsDistanceType().value;
    } else {
      metric =
          switch (params.getMetric()) {
            case L2 -> 0;
            case INNER_PRODUCT -> 1;
            default ->
                throw new IllegalArgumentException("Unsupported metric: " + params.getMetric());
          };
    }
    cuvsTieredIndexParams.metric(seg, metric);
    int algo = 0; // CUVS_TIERED_INDEX_ALGO_CAGRA
    cuvsTieredIndexParams.algo(seg, algo);
    cuvsTieredIndexParams.min_ann_rows(seg, params.getMinAnnRows());
    cuvsTieredIndexParams.create_ann_index_on_extend(seg, params.isCreateAnnIndexOnExtend());

    CagraIndexParams cagraParams = params.getCagraParams();
    if (cagraParams != null) {
      MemorySegment cagraParamsSeg = cuvsCagraIndexParams.allocate(arena);
      cuvsCagraIndexParams.intermediate_graph_degree(
          cagraParamsSeg, cagraParams.getIntermediateGraphDegree());
      cuvsCagraIndexParams.graph_degree(cagraParamsSeg, cagraParams.getGraphDegree());
      cuvsCagraIndexParams.build_algo(cagraParamsSeg, cagraParams.getCagraGraphBuildAlgo().value);
      cuvsCagraIndexParams.nn_descent_niter(
          cagraParamsSeg, cagraParams.getNNDescentNumIterations());
      cuvsCagraIndexParams.metric(cagraParamsSeg, metric);
      cuvsTieredIndexParams.cagra_params(seg, cagraParamsSeg);
    }
    cuvsTieredIndexParams.ivf_flat_params(seg, MemorySegment.NULL);
    cuvsTieredIndexParams.ivf_pq_params(seg, MemorySegment.NULL);

    return paramsHandle;
  }

  /**
   * Allocates the configured search parameters in the MemorySegment.
   */
  private MemorySegment segmentFromSearchParams(CagraSearchParams params, Arena arena) {
    MemorySegment seg = cuvsCagraSearchParams.allocate(arena);
    cuvsCagraSearchParams.max_queries(seg, params.getMaxQueries());
    cuvsCagraSearchParams.itopk_size(seg, params.getITopKSize());
    cuvsCagraSearchParams.max_iterations(seg, params.getMaxIterations());
    if (params.getCagraSearchAlgo() != null) {
      cuvsCagraSearchParams.algo(seg, params.getCagraSearchAlgo().value);
    }
    cuvsCagraSearchParams.team_size(seg, params.getTeamSize());
    cuvsCagraSearchParams.search_width(seg, params.getSearchWidth());
    cuvsCagraSearchParams.min_iterations(seg, params.getMinIterations());
    cuvsCagraSearchParams.thread_block_size(seg, params.getThreadBlockSize());
    if (params.getHashMapMode() != null) {
      cuvsCagraSearchParams.hashmap_mode(seg, params.getHashMapMode().value);
    }
    cuvsCagraSearchParams.hashmap_max_fill_rate(seg, params.getHashMapMaxFillRate());
    cuvsCagraSearchParams.num_random_samplings(seg, params.getNumRandomSamplings());
    cuvsCagraSearchParams.rand_xor_mask(seg, params.getRandXORMask());
    return seg;
  }

  /**
   * Gets an instance of {@link CuVSResources}
   *
   * @return an instance of {@link CuVSResources}
   */
  @Override
  public CuVSResources getCuVSResources() {
    return resources;
  }

  /**
   * Gets the index type
   *
   * @return the index type
   */
  @Override
  public TieredIndexType getIndexType() {
    TieredIndexType indexType = TieredIndexType.CAGRA; // Default to CAGRA for now
    return indexType;
  }

  /**
   * Static method to create a new builder
   */
  public static TieredIndex.Builder newBuilder(CuVSResources cuvsResources) {
    Objects.requireNonNull(cuvsResources);
    return new TieredIndexImpl.Builder(cuvsResources);
  }

  /**
   * Builder helps configure and create an instance of {@link TieredIndex}.
   */
  public static class Builder implements TieredIndex.Builder {
    private final CuVSResources resources;
    private CuVSMatrix dataset;
    private TieredIndexParams params;
    private TieredIndexType indexType = TieredIndexType.CAGRA;
    private InputStream inputStream;

    private Builder(CuVSResources resources) {
      this.resources = resources;
    }

    @Override
    public Builder from(InputStream inputStream) {
      this.inputStream = inputStream;
      return this;
    }

    @Override
    public Builder withDataset(float[][] vectors) {
      if (this.dataset != null) {
        throw new IllegalArgumentException("An input dataset can only be specified once");
      }
      if (vectors == null || vectors.length == 0 || vectors[0].length == 0) {
        throw new IllegalArgumentException("The input vectors cannot be null or empty");
      }
      this.dataset = CuVSMatrix.ofArray(vectors);
      return this;
    }

    @Override
    public Builder withDataset(CuVSMatrix dataset) {
      if (this.dataset != null) {
        throw new IllegalArgumentException("An input dataset can only be specified once");
      }
      if (dataset == null) {
        throw new IllegalArgumentException("An input dataset cannot be null");
      }
      this.dataset = dataset;
      return this;
    }

    @Override
    public Builder withIndexParams(TieredIndexParams params) {
      this.params = params;
      return this;
    }

    @Override
    public Builder withIndexType(TieredIndexType indexType) {
      this.indexType = indexType;
      return this;
    }

    @Override
    public TieredIndex build() throws Throwable {
      if (inputStream != null) {
        return new TieredIndexImpl(inputStream, resources);
      } else {
        if (dataset == null) {
          throw new IllegalArgumentException("Must provide a dataset");
        }
        if (params == null) {
          throw new IllegalStateException("Index parameters must be provided");
        }
        return new TieredIndexImpl(params, dataset, resources);
      }
    }
  }

  /**
   * Holds the memory reference to a Tiered index.
   */
  public static class IndexReference {
    private final MemorySegment memorySegment;

    /**
     * Constructs TieredIndexReference with an instance of MemorySegment passed as a
     * parameter.
     */
    protected IndexReference(MemorySegment indexMemorySegment) {
      this.memorySegment = indexMemorySegment;
    }

    /**
     * Gets the instance of index MemorySegment.
     */
    protected MemorySegment getMemorySegment() {
      return memorySegment;
    }
  }
}

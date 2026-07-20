/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal;

import static com.nvidia.cuvs.internal.CuVSParamsHelper.*;
import static com.nvidia.cuvs.internal.common.CloseableRMMAllocation.allocateRMMSegment;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_INT;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_INT_BYTE_SIZE;
import static com.nvidia.cuvs.internal.common.Util.CudaMemcpyKind.DEVICE_TO_HOST;
import static com.nvidia.cuvs.internal.common.Util.CudaMemcpyKind.HOST_TO_DEVICE;
import static com.nvidia.cuvs.internal.common.Util.buildMemorySegment;
import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.common.Util.concatenate;
import static com.nvidia.cuvs.internal.common.Util.prepareTensor;
import static com.nvidia.cuvs.internal.panama.headers_h.*;

import com.nvidia.cuvs.*;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.internal.common.CloseableHandle;
import com.nvidia.cuvs.internal.common.CloseableRMMAllocation;
import com.nvidia.cuvs.internal.common.CompositeCloseableHandle;
import com.nvidia.cuvs.internal.common.Util;
import com.nvidia.cuvs.internal.panama.*;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * {@link CagraIndex} encapsulates a CAGRA index, along with methods to interact
 * with it.
 * <p>
 * CAGRA is a graph-based nearest neighbors algorithm that was built from the
 * ground up for GPU acceleration. CAGRA demonstrates state-of-the art index
 * build and query performance for both small and large-batch sized search. Know
 * more about this algorithm
 * <a href="https://arxiv.org/abs/2308.15136" target="_blank">here</a>
 *
 * @since 25.02
 */
public class CagraIndexImpl implements CagraIndex {
  private final CuVSResources resources;
  private final IndexReference cagraIndexReference;
  private boolean destroyed;

  /**
   * Constructor for building the index using specified dataset
   *
   * @param indexParameters an instance of {@link CagraIndexParams} holding the
   *                        index parameters
   * @param dataset         the dataset for indexing
   * @param resources       an instance of {@link CuVSResources}
   */
  private CagraIndexImpl(
      CagraIndexParams indexParameters, CuVSMatrix dataset, CuVSResources resources) {
    Objects.requireNonNull(dataset);
    this.resources = resources;
    assert dataset instanceof CuVSMatrixInternal;
    this.cagraIndexReference = build(indexParameters, (CuVSMatrixInternal) dataset);
  }

  /**
   * Constructor for loading the index from an {@link InputStream}
   *
   * @param inputStream an instance of stream to read the index bytes from
   * @param resources   an instance of {@link CuVSResources}
   */
  private CagraIndexImpl(InputStream inputStream, CuVSResources resources) throws Throwable {
    this.resources = resources;
    this.cagraIndexReference = deserialize(inputStream);
  }

  /**
   * Constructor for creating an index from an existing index reference.
   * Used primarily for the merge operation.
   *
   * @param indexReference The reference to the existing index
   * @param resources The resources instance
   */
  private CagraIndexImpl(IndexReference indexReference, CuVSResources resources) {
    this.resources = resources;
    this.cagraIndexReference = indexReference;
    this.destroyed = false;
  }

  /**
   * Constructor for creating an index from a pre-build CAGRA graph
   *
   * @param metric      the distance type used
   * @param graph       a previously built CAGRA graph
   * @param dataset     the dataset used for indexing
   * @param resources   an instance of {@link CuVSResources}
   */
  private CagraIndexImpl(
      CagraIndexParams.CuvsDistanceType metric,
      CuVSMatrix graph,
      CuVSMatrix dataset,
      CuVSResources resources) {
    Objects.requireNonNull(graph);
    Objects.requireNonNull(dataset);

    this.resources = resources;

    assert graph instanceof CuVSMatrixInternal;
    assert dataset instanceof CuVSMatrixInternal;

    this.cagraIndexReference =
        fromGraph(metric, (CuVSMatrixInternal) graph, (CuVSMatrixInternal) dataset);
  }

  private void checkNotDestroyed() {
    if (destroyed) {
      throw new IllegalStateException("destroyed");
    }
  }

  /**
   * Invokes the native destroy_cagra_index to de-allocate the CAGRA index
   */
  @Override
  public void close() {
    checkNotDestroyed();
    try {
      int returnValue = cuvsCagraIndexDestroy(cagraIndexReference.getMemorySegment());
      checkCuVSError(returnValue, "cuvsCagraIndexDestroy");
      if (cagraIndexReference.dataset != null) {
        cagraIndexReference.dataset.close();
      }
    } finally {
      destroyed = true;
    }
  }

  /**
   * Invokes the native build_cagra_index function via the Panama API to build the
   * {@link CagraIndex}
   *
   * @return an instance of {@link IndexReference} that holds the pointer to the
   *         index
   */
  private IndexReference build(CagraIndexParams indexParameters, CuVSMatrixInternal dataset) {
    long rows = dataset.size();

    try (var indexParams = segmentFromIndexParams(indexParameters);
        var localArena = Arena.ofConfined()) {
      MemorySegment indexParamsMemorySegment = indexParams.handle();

      int numWriterThreads = indexParameters != null ? indexParameters.getNumWriterThreads() : 1;
      omp_set_num_threads(numWriterThreads);

      var datasetTensor = dataset.toTensor(localArena);
      var index = createCagraIndex();

      if (cuvsCagraIndexParams.build_algo(indexParamsMemorySegment)
          == 1) { // when build algo is IVF_PQ
        MemorySegment cuvsIvfPqIndexParamsMS =
            cuvsIvfPqParams.ivf_pq_build_params(
                cuvsCagraIndexParams.graph_build_params(indexParamsMemorySegment));
        int n_lists = cuvsIvfPqIndexParams.n_lists(cuvsIvfPqIndexParamsMS);
        // As rows cannot be less than n_lists value so trim down.
        cuvsIvfPqIndexParams.n_lists(
            cuvsIvfPqIndexParamsMS, (int) (rows < n_lists ? rows : n_lists));
      }
      try (var resourcesAccessor = resources.access()) {
        var cuvsRes = resourcesAccessor.handle();

        // TODO: do we need a stream sync here?
        var returnValue = cuvsStreamSync(cuvsRes);
        checkCuVSError(returnValue, "cuvsStreamSync");

        var viewKindSeg = localArena.allocate(C_INT);
        returnValue = cuvsCagraGetDatasetViewKind(datasetTensor, viewKindSeg);
        checkCuVSError(returnValue, "cuvsCagraGetDatasetViewKind");
        int viewKind = viewKindSeg.get(C_INT, 0);

        MemorySegment paddedView = MemorySegment.NULL;
        MemorySegment standardView = MemorySegment.NULL;
        try {
          switch (viewKind) {
            case 0 -> { // CUVS_DATASET_VIEW_KIND_DEVICE_PADDED
              MemorySegment paddedViewPtr = localArena.allocate(cuvsDatasetPaddedView_t);
              returnValue = cuvsDatasetMakeDevicePaddedView(cuvsRes, datasetTensor, paddedViewPtr);
              checkCuVSError(returnValue, "cuvsDatasetMakeDevicePaddedView");
              paddedView = paddedViewPtr.get(cuvsDatasetPaddedView_t, 0);
              returnValue =
                  cuvsCagraBuildDevicePadded(cuvsRes, indexParamsMemorySegment, paddedView, index);
              checkCuVSError(returnValue, "cuvsCagraBuildDevicePadded");
            }
            case 1 -> { // CUVS_DATASET_VIEW_KIND_HOST_PADDED
              MemorySegment paddedViewPtr = localArena.allocate(cuvsDatasetPaddedView_t);
              returnValue = cuvsDatasetMakeHostPaddedView(cuvsRes, datasetTensor, paddedViewPtr);
              checkCuVSError(returnValue, "cuvsDatasetMakeHostPaddedView");
              paddedView = paddedViewPtr.get(cuvsDatasetPaddedView_t, 0);
              returnValue =
                  cuvsCagraBuildHostPadded(cuvsRes, indexParamsMemorySegment, paddedView, index);
              checkCuVSError(returnValue, "cuvsCagraBuildHostPadded");
            }
            case 2 -> { // CUVS_DATASET_VIEW_KIND_DEVICE_STANDARD
              MemorySegment standardViewPtr = localArena.allocate(cuvsDatasetStandardView_t);
              returnValue =
                  cuvsDatasetMakeDeviceStandardView(cuvsRes, datasetTensor, standardViewPtr);
              checkCuVSError(returnValue, "cuvsDatasetMakeDeviceStandardView");
              standardView = standardViewPtr.get(cuvsDatasetStandardView_t, 0);
              returnValue =
                  cuvsCagraBuildDeviceStandard(
                      cuvsRes, indexParamsMemorySegment, standardView, index);
              checkCuVSError(returnValue, "cuvsCagraBuildDeviceStandard");
            }
            case 3 -> { // CUVS_DATASET_VIEW_KIND_HOST_STANDARD
              MemorySegment standardViewPtr = localArena.allocate(cuvsDatasetStandardView_t);
              returnValue =
                  cuvsDatasetMakeHostStandardView(cuvsRes, datasetTensor, standardViewPtr);
              checkCuVSError(returnValue, "cuvsDatasetMakeHostStandardView");
              standardView = standardViewPtr.get(cuvsDatasetStandardView_t, 0);
              returnValue =
                  cuvsCagraBuildHostStandard(
                      cuvsRes, indexParamsMemorySegment, standardView, index);
              checkCuVSError(returnValue, "cuvsCagraBuildHostStandard");
            }
            default ->
                throw new IllegalStateException("Unsupported CAGRA dataset view kind: " + viewKind);
          }
        } finally {
          if (paddedView.address() != 0) {
            checkCuVSError(
                cuvsDatasetPaddedViewDestroy(paddedView), "cuvsDatasetPaddedViewDestroy");
          }
          if (standardView.address() != 0) {
            checkCuVSError(
                cuvsDatasetStandardViewDestroy(standardView), "cuvsDatasetStandardViewDestroy");
          }
        }

        returnValue = cuvsStreamSync(cuvsRes);
        checkCuVSError(returnValue, "cuvsStreamSync");
      }
      omp_set_num_threads(1);

      return new IndexReference(index, dataset);
    }
  }

  private static MemorySegment createCagraIndex() {
    try (var localArena = Arena.ofConfined()) {
      MemorySegment indexPtrPtr = localArena.allocate(cuvsCagraIndex_t);
      // cuvsCagraIndexCreate gets a pointer to a cuvsCagraIndex_t, which is defined as a pointer to
      // cuvsCagraIndex.
      // It's basically an "out" parameter: the C functions will create the index and "return back"
      // a pointer to it: (*index = new cuvsCagraIndex{};
      // The "out parameter" pointer is needed only for the duration of the function invocation (it
      // could be a stack pointer, in C) so we allocate it from our localArena
      var returnValue = cuvsCagraIndexCreate(indexPtrPtr);
      checkCuVSError(returnValue, "cuvsCagraIndexCreate");
      return indexPtrPtr.get(cuvsCagraIndex_t, 0);
    }
  }

  private static final BitSet[] EMPTY_PREFILTER_BITSET = new BitSet[0];

  /**
   * Invokes the native search_cagra_index via the Panama API for searching a
   * CAGRA index.
   *
   * @param query an instance of {@link CagraQuery} holding the query vectors and
   *              other parameters
   * @return an instance of {@link CagraSearchResults} containing the results
   */
  @Override
  public SearchResults search(CagraQuery query) throws Throwable {
    try (var localArena = Arena.ofConfined()) {
      checkNotDestroyed();
      int topK = query.getTopK();
      var queryVectors = (CuVSMatrixInternal) query.getQueryVectors();
      long numQueries = queryVectors.size();
      long numBlocks = topK * numQueries;

      SequenceLayout neighborsSequenceLayout = MemoryLayout.sequenceLayout(numBlocks, C_INT);
      SequenceLayout distancesSequenceLayout =
          MemoryLayout.sequenceLayout(numBlocks, queryVectors.valueLayout());
      MemorySegment neighborsMemorySegment = localArena.allocate(neighborsSequenceLayout);
      MemorySegment distancesMemorySegment = localArena.allocate(distancesSequenceLayout);

      final long neighborsBytes = C_INT_BYTE_SIZE * numQueries * topK;
      final long distancesBytes = queryVectors.valueLayout().byteSize() * numQueries * topK;
      final boolean hasPreFilter = query.getPrefilter() != null;
      final BitSet[] prefilters =
          hasPreFilter ? new BitSet[] {query.getPrefilter()} : EMPTY_PREFILTER_BITSET;
      final long prefilterDataLength = hasPreFilter ? query.getNumDocs() * prefilters.length : 0;
      final long prefilterLen = hasPreFilter ? (prefilterDataLength + 31) / 32 : 0;
      final long prefilterBytes = C_INT_BYTE_SIZE * prefilterLen;

      try (var resourcesAccessor = query.getResources().access()) {
        var cuvsRes = resourcesAccessor.handle();
        var cuvsStream = Util.getStream(cuvsRes);

        try (var deviceQueryVectors =
                (CuVSMatrixInternal) queryVectors.toDevice(query.getResources());
            var neighborsDP = allocateRMMSegment(cuvsRes, neighborsBytes);
            var distancesDP = allocateRMMSegment(cuvsRes, distancesBytes);
            var prefilterDP =
                hasPreFilter
                    ? allocateRMMSegment(cuvsRes, prefilterBytes)
                    : CloseableRMMAllocation.EMPTY) {

          var queryTensor = deviceQueryVectors.toTensor(localArena);
          long[] neighborsShape = {numQueries, topK};
          MemorySegment neighborsTensor =
              prepareTensor(
                  localArena, neighborsDP.handle(), neighborsShape, kDLUInt(), 32, kDLCUDA());
          long[] distancesShape = {numQueries, topK};
          MemorySegment distancesTensor =
              prepareTensor(
                  localArena,
                  distancesDP.handle(),
                  distancesShape,
                  deviceQueryVectors.code(),
                  deviceQueryVectors.bits(),
                  kDLCUDA());

          // prepare the prefiltering data
          MemorySegment prefilter = cuvsFilter.allocate(localArena);

          if (!hasPreFilter) {
            cuvsFilter.type(prefilter, 0); // NO_FILTER
            cuvsFilter.addr(prefilter, 0);
          } else {
            BitSet concatenatedFilters = concatenate(prefilters, query.getNumDocs());
            long[] filters = concatenatedFilters.toLongArray();
            var prefilterDataMemorySegment = buildMemorySegment(localArena, filters);

            long[] prefilterShape = {prefilterLen};

            Util.cudaMemcpyAsync(
                prefilterDP.handle(),
                prefilterDataMemorySegment,
                prefilterBytes,
                HOST_TO_DEVICE,
                cuvsStream);

            MemorySegment prefilterTensor =
                prepareTensor(
                    localArena, prefilterDP.handle(), prefilterShape, kDLUInt(), 32, kDLCUDA());

            cuvsFilter.type(prefilter, 1);
            cuvsFilter.addr(prefilter, prefilterTensor.address());
          }

          // TODO: do we need this stream sync here?
          checkCuVSError(cuvsStreamSync(cuvsRes), "cuvsStreamSync");

          var returnValue =
              cuvsCagraSearch(
                  cuvsRes,
                  segmentFromSearchParams(localArena, query.getCagraSearchParameters()),
                  cagraIndexReference.getMemorySegment(),
                  queryTensor,
                  neighborsTensor,
                  distancesTensor,
                  prefilter);
          checkCuVSError(returnValue, "cuvsCagraSearch");

          // TODO: we can avoid/defer this using CuVSDeviceMatrix for neighborsDP and distancesDP
          // TODO: also, should we use cuvsMatrixCopy instead?
          Util.cudaMemcpyAsync(
              neighborsMemorySegment,
              neighborsDP.handle(),
              neighborsBytes,
              DEVICE_TO_HOST,
              cuvsStream);
          Util.cudaMemcpyAsync(
              distancesMemorySegment,
              distancesDP.handle(),
              distancesBytes,
              DEVICE_TO_HOST,
              cuvsStream);

          checkCuVSError(cuvsStreamSync(cuvsRes), "cuvsStreamSync");
        }
      }

      return CagraSearchResults.create(
          neighborsSequenceLayout,
          distancesSequenceLayout,
          neighborsMemorySegment,
          distancesMemorySegment,
          topK,
          query.getMapping(),
          numQueries);
    }
  }

  @Override
  public void serialize(OutputStream outputStream) throws Throwable {
    Path path =
        Files.createTempFile(resources.tempDirectory(), UUID.randomUUID().toString(), ".cag");
    serialize(outputStream, path, 1024);
  }

  @Override
  public void serialize(OutputStream outputStream, int bufferLength) throws Throwable {
    Path path =
        Files.createTempFile(resources.tempDirectory(), UUID.randomUUID().toString(), ".cag");
    serialize(outputStream, path, bufferLength);
  }

  @Override
  public void serialize(OutputStream outputStream, Path tempFile, int bufferLength)
      throws Throwable {
    checkNotDestroyed();
    final var tempFilePath = tempFile.toAbsolutePath();
    try (var localArena = Arena.ofConfined();
        var resourcesAccessor = resources.access()) {

      long cuvsRes = resourcesAccessor.handle();
      var returnValue =
          cuvsCagraSerialize(
              cuvsRes,
              localArena.allocateFrom(tempFilePath.toString()),
              cagraIndexReference.getMemorySegment(),
              true);
      checkCuVSError(returnValue, "cuvsCagraSerialize");

      try (var fileInputStream = Files.newInputStream(tempFilePath)) {
        byte[] chunk = new byte[bufferLength];
        int chunkLength = 0;
        while ((chunkLength = fileInputStream.read(chunk)) != -1) {
          outputStream.write(chunk, 0, chunkLength);
        }
      } finally {
        Files.deleteIfExists(tempFilePath);
      }
    }
  }

  @Override
  public CuVSDeviceMatrix getGraph() {
    try (var localArena = Arena.ofConfined()) {
      // Use a "device" graph + tensor, avoid (defer) copy
      MemorySegment graphDeviceTensor = DLManagedTensor.allocate(localArena);
      DLManagedTensor.dl_tensor(graphDeviceTensor, DLTensor.allocate(localArena));

      checkCuVSError(
          cuvsCagraIndexGetGraph(cagraIndexReference.getMemorySegment(), graphDeviceTensor),
          "cuvsCagraIndexGetGraph");

      assert DLTensor.ndim(DLManagedTensor.dl_tensor(graphDeviceTensor)) == 2;
      assert DLTensor.shape(DLManagedTensor.dl_tensor(graphDeviceTensor)).get(int64_t, 0) > 0;
      assert DLTensor.shape(DLManagedTensor.dl_tensor(graphDeviceTensor)).getAtIndex(int64_t, 1)
          > 0;

      var graph = CuVSMatrixBaseImpl.fromTensor(graphDeviceTensor, resources);
      assert graph instanceof CuVSDeviceMatrix;
      return (CuVSDeviceMatrix) graph;
    }
  }

  private IndexReference fromGraph(
      CagraIndexParams.CuvsDistanceType metric,
      CuVSMatrixInternal graph,
      CuVSMatrixInternal dataset) {
    try (var localArena = Arena.ofConfined()) {
      var index = createCagraIndex();
      try (var resourcesAccess = resources.access()) {
        long cuvsRes = resourcesAccess.handle();

        MemorySegment datasetTensor = dataset.toTensor(localArena);
        MemorySegment graphTensor = graph.toTensor(localArena);

        checkCuVSError(
            cuvsCagraIndexFromArgs(cuvsRes, metric.value, graphTensor, datasetTensor, index),
            "cuvsCagraIndexFromArgs");
      }
      return new IndexReference(index, dataset);
    }
  }

  @Override
  public void serializeToHNSW(OutputStream outputStream) throws Throwable {
    Path path =
        Files.createTempFile(resources.tempDirectory(), UUID.randomUUID().toString(), ".hnsw");
    serializeToHNSW(outputStream, path, 1024);
  }

  @Override
  public void serializeToHNSW(OutputStream outputStream, int bufferLength) throws Throwable {
    Path path =
        Files.createTempFile(resources.tempDirectory(), UUID.randomUUID().toString(), ".hnsw");
    serializeToHNSW(outputStream, path, bufferLength);
  }

  @Override
  public void serializeToHNSW(OutputStream outputStream, Path tempFile, int bufferLength)
      throws Throwable {
    checkNotDestroyed();
    final var tempFilePath = tempFile.toAbsolutePath();

    try (var localArena = Arena.ofConfined()) {
      MemorySegment pathSeg = buildMemorySegment(localArena, tempFile.toString());

      try (var resourcesAccessor = resources.access()) {
        checkCuVSError(
            cuvsCagraSerializeToHnswlib(
                resourcesAccessor.handle(), pathSeg, cagraIndexReference.getMemorySegment()),
            "cuvsCagraSerializeToHnswlib");
      }
    }

    try (FileInputStream fileInputStream = new FileInputStream(tempFilePath.toFile())) {
      byte[] chunk = new byte[bufferLength];
      int chunkLength;
      while ((chunkLength = fileInputStream.read(chunk)) != -1) {
        outputStream.write(chunk, 0, chunkLength);
      }
    } finally {
      Files.deleteIfExists(tempFilePath);
    }
  }

  /**
   * Gets an instance of {@link IndexReference} by deserializing a CAGRA index
   * using an {@link InputStream}.
   *
   * @param inputStream  an instance of {@link InputStream}
   * @return an instance of {@link IndexReference}.
   */
  private IndexReference deserialize(InputStream inputStream) throws Throwable {
    Path tmpIndexFile =
        Files.createTempFile(resources.tempDirectory(), UUID.randomUUID().toString(), ".cag")
            .toAbsolutePath();
    var index = createCagraIndex();

    try (inputStream;
        var outputStream = Files.newOutputStream(tmpIndexFile);
        var arena = Arena.ofConfined()) {
      inputStream.transferTo(outputStream);

      try (var resourcesAccessor = resources.access()) {
        var cuvsRes = resourcesAccessor.handle();
        var returnValue =
            cuvsCagraDeserialize(
                cuvsRes,
                arena.allocateFrom(tmpIndexFile.toString()),
                0, // CUVS_DATASET_LAYOUT_STANDARD
                index);
        checkCuVSError(returnValue, "cuvsCagraDeserialize");
      }
    } finally {
      Files.deleteIfExists(tmpIndexFile);
    }
    return new IndexReference(index, null);
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
   * Gets the CAGRA index reference (for internal use).
   * Package-private to allow access from HnswIndexImpl.
   *
   * @return the memory segment representing the CAGRA index
   */
  MemorySegment getCagraIndexReference() {
    return cagraIndexReference.getMemorySegment();
  }

  CuVSMatrix getDatasetForConversion() {
    return cagraIndexReference.dataset;
  }

  /**
   * Allocates the native CagraIndexParams data structures and fills the configured index parameters in.
   */
  private static CloseableHandle segmentFromIndexParams(CagraIndexParams params) {
    var handles = new ArrayList<CloseableHandle>();

    var indexParams = createCagraIndexParams();
    handles.add(indexParams);
    var indexPtr = indexParams.handle();

    if (params != null) {
      populateNativeIndexParams(indexPtr, params, handles);
    }

    return new CompositeCloseableHandle(indexPtr, handles);
  }

  private static void populateNativeIndexParams(
      MemorySegment indexPtr, CagraIndexParams params, List<CloseableHandle> handles) {

    cuvsCagraIndexParams.intermediate_graph_degree(indexPtr, params.getIntermediateGraphDegree());
    cuvsCagraIndexParams.graph_degree(indexPtr, params.getGraphDegree());
    cuvsCagraIndexParams.build_algo(indexPtr, params.getCagraGraphBuildAlgo().value);
    cuvsCagraIndexParams.nn_descent_niter(indexPtr, params.getNNDescentNumIterations());
    cuvsCagraIndexParams.metric(indexPtr, params.getCuvsDistanceType().value);

    if (params.getCagraGraphBuildAlgo().equals(CagraGraphBuildAlgo.IVF_PQ)) {

      var ivfPqIndexParams = createIvfPqIndexParams();
      handles.add(ivfPqIndexParams);
      MemorySegment ivfpqIndexParamsMemorySegment = ivfPqIndexParams.handle();
      CuVSIvfPqIndexParams cuVSIvfPqIndexParams = params.getCuVSIvfPqParams().getIndexParams();

      cuvsIvfPqIndexParams.metric(
          ivfpqIndexParamsMemorySegment, cuVSIvfPqIndexParams.getMetric().value);
      cuvsIvfPqIndexParams.metric_arg(
          ivfpqIndexParamsMemorySegment, cuVSIvfPqIndexParams.getMetricArg());
      cuvsIvfPqIndexParams.add_data_on_build(
          ivfpqIndexParamsMemorySegment, cuVSIvfPqIndexParams.isAddDataOnBuild());
      cuvsIvfPqIndexParams.n_lists(ivfpqIndexParamsMemorySegment, cuVSIvfPqIndexParams.getnLists());
      cuvsIvfPqIndexParams.kmeans_n_iters(
          ivfpqIndexParamsMemorySegment, cuVSIvfPqIndexParams.getKmeansNIters());
      cuvsIvfPqIndexParams.kmeans_trainset_fraction(
          ivfpqIndexParamsMemorySegment, cuVSIvfPqIndexParams.getKmeansTrainsetFraction());
      cuvsIvfPqIndexParams.pq_bits(ivfpqIndexParamsMemorySegment, cuVSIvfPqIndexParams.getPqBits());
      cuvsIvfPqIndexParams.pq_dim(ivfpqIndexParamsMemorySegment, cuVSIvfPqIndexParams.getPqDim());
      cuvsIvfPqIndexParams.codebook_kind(
          ivfpqIndexParamsMemorySegment, cuVSIvfPqIndexParams.getCodebookKind().value);
      cuvsIvfPqIndexParams.force_random_rotation(
          ivfpqIndexParamsMemorySegment, cuVSIvfPqIndexParams.isForceRandomRotation());
      cuvsIvfPqIndexParams.conservative_memory_allocation(
          ivfpqIndexParamsMemorySegment, cuVSIvfPqIndexParams.isConservativeMemoryAllocation());
      cuvsIvfPqIndexParams.max_train_points_per_pq_code(
          ivfpqIndexParamsMemorySegment, cuVSIvfPqIndexParams.getMaxTrainPointsPerPqCode());

      var ivfPqSearchParams = createIvfPqSearchParams();
      handles.add(ivfPqSearchParams);
      MemorySegment ivfpqSearchParamsMemorySegment = ivfPqSearchParams.handle();
      CuVSIvfPqSearchParams cuVSIvfPqSearchParams = params.getCuVSIvfPqParams().getSearchParams();
      cuvsIvfPqSearchParams.n_probes(
          ivfpqSearchParamsMemorySegment, cuVSIvfPqSearchParams.getnProbes());
      cuvsIvfPqSearchParams.lut_dtype(
          ivfpqSearchParamsMemorySegment, cuVSIvfPqSearchParams.getLutDtype().value);
      cuvsIvfPqSearchParams.internal_distance_dtype(
          ivfpqSearchParamsMemorySegment, cuVSIvfPqSearchParams.getInternalDistanceDtype().value);
      cuvsIvfPqSearchParams.preferred_shmem_carveout(
          ivfpqSearchParamsMemorySegment, cuVSIvfPqSearchParams.getPreferredShmemCarveout());

      // This is already allocated by cuvsCagraIndexParamsCreate,
      // we just need to populate it.
      MemorySegment cuvsIvfPqParamsMemorySegment =
          cuvsCagraIndexParams.graph_build_params(indexPtr);
      cuvsIvfPqParams.ivf_pq_build_params(
          cuvsIvfPqParamsMemorySegment, ivfpqIndexParamsMemorySegment);
      cuvsIvfPqParams.ivf_pq_search_params(
          cuvsIvfPqParamsMemorySegment, ivfpqSearchParamsMemorySegment);
      cuvsIvfPqParams.refinement_rate(
          cuvsIvfPqParamsMemorySegment, params.getCuVSIvfPqParams().getRefinementRate());

      cuvsCagraIndexParams.graph_build_params(indexPtr, cuvsIvfPqParamsMemorySegment);
    } else if (params.getCagraGraphBuildAlgo().equals(CagraGraphBuildAlgo.ACE)) {
      var aceParams = createAceParams();
      // Note: Do NOT add aceParams to handles list.
      // The cuvsCagraIndexParamsDestroy will handle freeing the ACE params
      // when graph_build_algo is ACE, just like it does for IVF-PQ params.
      MemorySegment cuvsAceParamsMemorySegment = aceParams.handle();
      CuVSAceParams cuVSAceParams = params.getCuVSAceParams();

      cuvsAceParams.npartitions(cuvsAceParamsMemorySegment, cuVSAceParams.getNpartitions());
      cuvsAceParams.ef_construction(cuvsAceParamsMemorySegment, cuVSAceParams.getEfConstruction());
      cuvsAceParams.use_disk(cuvsAceParamsMemorySegment, cuVSAceParams.isUseDisk());
      cuvsAceParams.max_host_memory_gb(
          cuvsAceParamsMemorySegment, cuVSAceParams.getMaxHostMemoryGb());
      cuvsAceParams.max_gpu_memory_gb(
          cuvsAceParamsMemorySegment, cuVSAceParams.getMaxGpuMemoryGb());

      String buildDir = cuVSAceParams.getBuildDir();
      if (buildDir != null && !buildDir.isEmpty()) {
        MemorySegment buildDirSegment = Util.duplicateNativeString(buildDir);
        cuvsAceParams.build_dir(cuvsAceParamsMemorySegment, buildDirSegment);
      }

      cuvsCagraIndexParams.graph_build_params(indexPtr, cuvsAceParamsMemorySegment);
    }
  }

  /**
   * Allocates the configured search parameters in the MemorySegment.
   */
  private MemorySegment segmentFromSearchParams(Arena arena, CagraSearchParams params) {
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

  public static CagraIndex.Builder newBuilder(CuVSResources cuvsResources) {
    return new CagraIndexImpl.Builder(Objects.requireNonNull(cuvsResources));
  }

  /**
   * Merges multiple CAGRA indexes into a single index.
   *
   * @param indexes Array of CAGRA indexes to merge
   * @return A new merged CAGRA index
   */
  public static CagraIndex merge(CagraIndex[] indexes) {
    return merge(indexes, null);
  }

  /**
   * Merges multiple CAGRA indexes into a single index with specified merge parameters.
   *
   * @param indexes Array of CAGRA indexes to merge
   * @param mergeParams Parameters to control the merge operation, or null to use defaults
   * @return A new merged CAGRA index
   */
  public static CagraIndex merge(CagraIndex[] indexes, CagraIndexParams mergeParams) {
    CuVSResources resources = indexes[0].getCuVSResources();
    var mergedIndex = createCagraIndex();

    try (var localArena = Arena.ofConfined()) {
      MemorySegment indexesSegment =
          localArena.allocate(indexes.length * ValueLayout.ADDRESS.byteSize());

      for (int i = 0; i < indexes.length; i++) {
        CagraIndexImpl indexImpl = (CagraIndexImpl) indexes[i];
        indexesSegment.setAtIndex(
            ValueLayout.ADDRESS, i, indexImpl.cagraIndexReference.getMemorySegment());
      }

      try (var nativeMergeParams = segmentFromIndexParams(mergeParams);
          var resourcesAccessor = resources.access()) {
        var cuvsRes = resourcesAccessor.handle();

        MemorySegment mergeFilter = cuvsFilter.allocate(localArena);
        cuvsFilter.type(mergeFilter, 0); // NO_FILTER
        cuvsFilter.addr(mergeFilter, 0);

        MemorySegment mergedStorage = MemorySegment.NULL;
        try {
          MemorySegment mergedStoragePtr = localArena.allocate(cuvsDatasetStorage_t);
          checkCuVSError(
              cuvsMakeMergedStorage(
                  cuvsRes, indexesSegment, indexes.length, mergeFilter, mergedStoragePtr),
              "cuvsMakeMergedStorage");
          mergedStorage = mergedStoragePtr.get(cuvsDatasetStorage_t, 0);

          checkCuVSError(
              cuvsCagraMerge(
                  cuvsRes,
                  nativeMergeParams.handle(),
                  indexesSegment,
                  indexes.length,
                  mergeFilter,
                  mergedStorage,
                  mergedIndex),
              "cuvsCagraMerge");
        } finally {
          if (mergedStorage.address() != 0) {
            checkCuVSError(cuvsDatasetStorageDestroy(mergedStorage), "cuvsDatasetStorageDestroy");
          }
        }
      }
    }

    return new CagraIndexImpl(new IndexReference(mergedIndex, null), resources);
  }

  /**
   * Builder helps configure and create an instance of {@link CagraIndex}.
   */
  public static class Builder implements CagraIndex.Builder {

    private CuVSMatrix dataset;
    private CagraIndexParams cagraIndexParams;
    private final CuVSResources cuvsResources;
    private InputStream inputStream;
    private CuVSMatrix graph;

    public Builder(CuVSResources cuvsResources) {
      this.cuvsResources = cuvsResources;
    }

    @Override
    public Builder from(InputStream inputStream) {
      this.inputStream = inputStream;
      return this;
    }

    @Override
    public Builder from(CuVSMatrix graph) {
      this.graph = graph;
      return this;
    }

    @Override
    public Builder withDataset(float[][] vectors) {
      this.dataset = CuVSMatrix.ofArray(vectors);
      return this;
    }

    @Override
    public Builder withDataset(CuVSMatrix dataset) {
      this.dataset = dataset;
      return this;
    }

    @Override
    public Builder withIndexParams(CagraIndexParams cagraIndexParameters) {
      this.cagraIndexParams = cagraIndexParameters;
      return this;
    }

    @Override
    public CagraIndexImpl build() throws Throwable {
      if (inputStream != null) {
        return new CagraIndexImpl(inputStream, cuvsResources);
      } else {
        if (graph != null) {
          if (cagraIndexParams == null || dataset == null) {
            throw new IllegalArgumentException(
                "In order to reconstruct a CAGRA index from a graph, "
                    + "you must specify the original dataset and the metric used.");
          }
          return new CagraIndexImpl(
              cagraIndexParams.getCuvsDistanceType(), graph, dataset, cuvsResources);
        } else {
          return new CagraIndexImpl(cagraIndexParams, dataset, cuvsResources);
        }
      }
    }
  }

  /**
   * Holds the memory reference to a CAGRA index.
   */
  public static class IndexReference {

    private final MemorySegment memorySegment;
    private final CuVSMatrix dataset;

    /**
     * Constructs CagraIndexReference with an instance of MemorySegment passed as a
     * parameter.
     *
     * @param indexMemorySegment the MemorySegment instance to use for containing
     *                           index reference
     * @param dataset            the dataset used for indexing; the dataset lifetime
     *                           matches the lifetime of the index, we need to keep a reference
     *                           to it so we can close it when the index is closed.
     *                           Can be null (e.g. from deserialization or merging)
     */
    private IndexReference(MemorySegment indexMemorySegment, CuVSMatrix dataset) {
      this.memorySegment = indexMemorySegment;
      this.dataset = dataset;
    }

    /**
     * Gets the instance of index MemorySegment.
     *
     * @return index MemorySegment
     */
    protected MemorySegment getMemorySegment() {
      return memorySegment;
    }
  }
}

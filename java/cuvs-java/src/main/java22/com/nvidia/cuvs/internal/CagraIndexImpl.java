/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal;

import static com.nvidia.cuvs.internal.CuVSParamsHelper.*;
import static com.nvidia.cuvs.internal.common.CloseableRMMAllocation.allocateRMMSegment;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_FLOAT;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_FLOAT_BYTE_SIZE;
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
    this(inputStream, resources, null);
  }

  private CagraIndexImpl(
      InputStream inputStream, CuVSResources resources, CagraIndex.DeserializeDataset outDataset)
      throws Throwable {
    this.resources = resources;
    this.cagraIndexReference = deserialize(inputStream, outDataset);
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
      if (cagraIndexReference.datasetOwner != null) {
        try {
          cagraIndexReference.datasetOwner.close();
        } catch (Exception e) {
          throw new RuntimeException("Failed to destroy CAGRA dataset", e);
        }
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

        MemorySegment datasetView = MemorySegment.NULL;
        try {
          MemorySegment datasetViewPtr = localArena.allocate(cuvsDataset_t);
          if (isCagraPaddedLayout(dataset)) {
            returnValue = cuvsDatasetMakePaddedView(cuvsRes, datasetTensor, datasetViewPtr);
            checkCuVSError(returnValue, "cuvsDatasetMakePaddedView");
          } else {
            returnValue = cuvsDatasetMakeStandardView(cuvsRes, datasetTensor, datasetViewPtr);
            checkCuVSError(returnValue, "cuvsDatasetMakeStandardView");
          }
          datasetView = datasetViewPtr.get(cuvsDataset_t, 0);

          returnValue = cuvsCagraBuild(cuvsRes, indexParamsMemorySegment, datasetView, index);
          checkCuVSError(returnValue, "cuvsCagraBuild");
        } finally {
          if (datasetView.address() != 0) {
            checkCuVSError(cuvsDatasetDestroy(datasetView), "cuvsDatasetDestroy");
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

  /** Matches C++ `cagra_required_row_width` (16-byte default alignment). */
  private static long cagraRequiredRowWidth(long logicalColumns, int sizeofValue) {
    int alignBytes = 16;
    int lcm = lcm(alignBytes, sizeofValue);
    long bytes = logicalColumns * (long) sizeofValue;
    long rounded = ((bytes + lcm - 1) / lcm) * lcm;
    return rounded / sizeofValue;
  }

  private static int lcm(int a, int b) {
    return a / gcd(a, b) * b;
  }

  private static int gcd(int a, int b) {
    while (b != 0) {
      int t = a % b;
      a = b;
      b = t;
    }
    return a;
  }

  private static int elementSizeBytes(CuVSMatrix.DataType dataType) {
    return switch (dataType) {
      case FLOAT, INT, UINT -> 4;
      case HALF -> 2;
      case BYTE -> 1;
    };
  }

  /**
   * True when the matrix row width matches CAGRA's required padded width for its logical column
   * count and element type. The stride lives on the internal matrix type, so this is the only place
   * that can answer the question; {@link com.nvidia.cuvs.CagraIndex#isPaddedDataset} routes here
   * through the provider.
   */
  public static boolean isPaddedDataset(CuVSMatrix dataset) {
    if (!(dataset instanceof CuVSMatrixInternal datasetInternal)) {
      throw new IllegalArgumentException("dataset must be a CuVSMatrixInternal matrix");
    }
    return isCagraPaddedLayout(datasetInternal);
  }

  /**
   * True when the matrix row width matches CAGRA's required padded width for its
   * logical column count and element type.
   */
  private static boolean isCagraPaddedLayout(CuVSMatrixInternal dataset) {
    long logicalColumns = dataset.columns();
    long rowStride = dataset.rowStride();
    long actualRowWidth = rowStride > 0 ? rowStride : logicalColumns;
    return actualRowWidth
        == cagraRequiredRowWidth(logicalColumns, elementSizeBytes(dataset.dataType()));
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
      SequenceLayout distancesSequenceLayout = MemoryLayout.sequenceLayout(numBlocks, C_FLOAT);
      MemorySegment neighborsMemorySegment = localArena.allocate(neighborsSequenceLayout);
      MemorySegment distancesMemorySegment = localArena.allocate(distancesSequenceLayout);

      final long neighborsBytes = C_INT_BYTE_SIZE * numQueries * topK;
      final long distancesBytes = C_FLOAT_BYTE_SIZE * numQueries * topK;
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
                  localArena, distancesDP.handle(), distancesShape, kDLFloat(), 32, kDLCUDA());

          // prepare the prefiltering data
          MemorySegment prefilter = cuvsFilter.allocate(localArena);

          if (!hasPreFilter) {
            cuvsFilter.type(prefilter, 0); // NO_FILTER
            cuvsFilter.addr(prefilter, 0);
          } else {
            BitSet concatenatedFilters = concatenate(prefilters, query.getNumDocs());
            long[] filters = concatenatedFilters.toLongArray();
            var prefilterDataMemorySegment =
                buildMemorySegment(localArena, filters, (prefilterDataLength + 63) / 64);

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

  /** Returns the underlying {@code cuvsCagraIndex_t} handle for native-side index passing. */
  public MemorySegment getIndexHandle() {
    return cagraIndexReference.getMemorySegment();
  }

  @Override
  public CagraIndex.PaddedDataset makePaddedDataset(CuVSMatrix dataset) throws Throwable {
    checkNotDestroyed();
    Objects.requireNonNull(dataset);
    if (!(dataset instanceof CuVSMatrixInternal datasetInternal)) {
      throw new IllegalArgumentException("dataset must be a CuVSMatrixInternal matrix");
    }

    try (var localArena = Arena.ofConfined();
        var resourcesAccessor = resources.access()) {
      var cuvsRes = resourcesAccessor.handle();
      var datasetTensor = datasetInternal.toTensor(localArena);
      int targetMemType =
          (datasetInternal instanceof CuVSHostMatrixImpl)
              ? CUVS_DATASET_MEM_TYPE_HOST()
              : CUVS_DATASET_MEM_TYPE_DEVICE();
      MemorySegment paddedDatasetPtr = localArena.allocate(cuvsDataset_t);
      var returnValue =
          cuvsDatasetMakePadded(cuvsRes, datasetTensor, targetMemType, paddedDatasetPtr);
      checkCuVSError(returnValue, "cuvsDatasetMakePadded");
      MemorySegment paddedDataset = paddedDatasetPtr.get(cuvsDataset_t, 0);

      var out = new CagraIndex.PaddedDataset();
      out.setDelegate(new DatasetCloseDelegate(paddedDataset), paddedDataset.address());
      return out;
    }
  }

  @Override
  public CagraIndex.PaddedDatasetView makePaddedDatasetView(CuVSMatrix dataset) throws Throwable {
    checkNotDestroyed();
    Objects.requireNonNull(dataset);
    if (!(dataset instanceof CuVSMatrixInternal datasetInternal)) {
      throw new IllegalArgumentException("dataset must be a CuVSMatrixInternal matrix");
    }

    try (var localArena = Arena.ofConfined();
        var resourcesAccessor = resources.access()) {
      var cuvsRes = resourcesAccessor.handle();
      var datasetTensor = datasetInternal.toTensor(localArena);
      MemorySegment paddedViewPtr = localArena.allocate(cuvsDataset_t);
      var returnValue = cuvsDatasetMakePaddedView(cuvsRes, datasetTensor, paddedViewPtr);
      checkCuVSError(returnValue, "cuvsDatasetMakePaddedView");
      MemorySegment paddedView = paddedViewPtr.get(cuvsDataset_t, 0);

      var out = new CagraIndex.PaddedDatasetView();
      out.setDelegate(new DatasetCloseDelegate(paddedView), paddedView.address());
      return out;
    }
  }

  @Override
  public CagraIndex.StandardDatasetView makeStandardDatasetView(CuVSMatrix dataset)
      throws Throwable {
    checkNotDestroyed();
    Objects.requireNonNull(dataset);
    if (!(dataset instanceof CuVSMatrixInternal datasetInternal)) {
      throw new IllegalArgumentException("dataset must be a CuVSMatrixInternal matrix");
    }

    try (var localArena = Arena.ofConfined();
        var resourcesAccessor = resources.access()) {
      var cuvsRes = resourcesAccessor.handle();
      var datasetTensor = datasetInternal.toTensor(localArena);
      MemorySegment standardViewPtr = localArena.allocate(cuvsDataset_t);
      var returnValue = cuvsDatasetMakeStandardView(cuvsRes, datasetTensor, standardViewPtr);
      checkCuVSError(returnValue, "cuvsDatasetMakeStandardView");
      MemorySegment standardView = standardViewPtr.get(cuvsDataset_t, 0);

      var out = new CagraIndex.StandardDatasetView();
      out.setDelegate(new DatasetCloseDelegate(standardView), standardView.address());
      return out;
    }
  }

  @Override
  public void updateDataset(CagraIndex.PaddedDatasetView datasetView) throws Throwable {
    checkNotDestroyed();
    Objects.requireNonNull(datasetView);
    if (!datasetView.isPresent()) {
      throw new IllegalArgumentException("datasetView is uninitialized");
    }
    updateDataset(datasetView.nativeHandleAddress());
  }

  @Override
  public void updateDataset(CagraIndex.PaddedDataset dataset) throws Throwable {
    checkNotDestroyed();
    Objects.requireNonNull(dataset);
    if (!dataset.isPresent()) {
      throw new IllegalArgumentException("dataset is uninitialized");
    }
    updateDataset(dataset.nativeHandleAddress());
  }

  private void updateDataset(long datasetHandleAddress) {
    try (var resourcesAccessor = resources.access()) {
      var cuvsRes = resourcesAccessor.handle();
      var returnValue =
          cuvsCagraUpdateDataset(
              cuvsRes,
              MemorySegment.ofAddress(datasetHandleAddress),
              cagraIndexReference.getMemorySegment());
      checkCuVSError(returnValue, "cuvsCagraUpdateDataset");
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
          cuvsCagraSerializeGraphAndDataset(
              cuvsRes,
              localArena.allocateFrom(tempFilePath.toString()),
              cagraIndexReference.getMemorySegment());
      checkCuVSError(returnValue, "cuvsCagraSerializeGraphAndDataset");

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

  @Override
  public long getGraphDegree() {
    try (var localArena = Arena.ofConfined()) {
      MemorySegment graphDegree = localArena.allocate(int64_t);
      checkCuVSError(
          cuvsCagraIndexGetGraphDegree(cagraIndexReference.getMemorySegment(), graphDegree),
          "cuvsCagraIndexGetGraphDegree");
      return graphDegree.get(int64_t, 0);
    }
  }

  @Override
  public long size() {
    checkNotDestroyed();
    try (var localArena = Arena.ofConfined()) {
      MemorySegment size = localArena.allocate(int64_t);
      checkCuVSError(
          cuvsCagraIndexGetSize(cagraIndexReference.getMemorySegment(), size),
          "cuvsCagraIndexGetSize");
      return size.get(int64_t, 0);
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
   * @param inputStream an instance of {@link InputStream}
   * @return an instance of {@link IndexReference}
   */
  private IndexReference deserialize(
      InputStream inputStream, CagraIndex.DeserializeDataset outDataset) throws Throwable {
    if (outDataset != null && outDataset.isPresent()) {
      throw new IllegalArgumentException("outDataset must be empty before deserialization");
    }
    if (outDataset != null
        && !(outDataset instanceof CagraIndex.PaddedDataset)
        && !(outDataset instanceof CagraIndex.StandardDataset)) {
      throw new IllegalArgumentException(
          "outDataset must be CagraIndex.PaddedDataset or CagraIndex.StandardDataset");
    }

    Path tmpIndexFile =
        Files.createTempFile(resources.tempDirectory(), UUID.randomUUID().toString(), ".cag")
            .toAbsolutePath();
    MemorySegment index = createCagraIndex();
    MemorySegment dataset = MemorySegment.NULL;

    try (inputStream;
        var outputStream = Files.newOutputStream(tmpIndexFile);
        var arena = Arena.ofConfined()) {
      inputStream.transferTo(outputStream);

      try (var resourcesAccessor = resources.access()) {
        MemorySegment datasetOutPtr = arena.allocate(cuvsDataset_t);
        datasetOutPtr.set(cuvsDataset_t, 0, MemorySegment.NULL);
        var returnValue =
            cuvsCagraDeserializeGraphAndDataset(
                resourcesAccessor.handle(),
                arena.allocateFrom(tmpIndexFile.toString()),
                index,
                datasetOutPtr);
        checkCuVSError(returnValue, "cuvsCagraDeserializeGraphAndDataset");
        dataset = datasetOutPtr.get(cuvsDataset_t, 0);
      }

      if (outDataset != null) {
        int expectedLayout =
            outDataset instanceof CagraIndex.PaddedDataset
                ? CUVS_DATASET_LAYOUT_PADDED()
                : CUVS_DATASET_LAYOUT_STANDARD();
        if (cuvsDataset.layout(dataset) != expectedLayout) {
          throw new IllegalArgumentException(
              "outDataset type does not match the serialized dataset layout");
        }
      }

      var datasetOwner = new DatasetCloseDelegate(dataset);
      if (outDataset == null) {
        dataset = MemorySegment.NULL;
        return new IndexReference(index, null, datasetOwner);
      }

      outDataset.setDelegate(datasetOwner, dataset.address());
      dataset = MemorySegment.NULL;
      return new IndexReference(index, null, null);
    } catch (Throwable t) {
      if (dataset.address() != 0) {
        try {
          checkCuVSError(cuvsDatasetDestroy(dataset), "cuvsDatasetDestroy");
        } catch (Throwable cleanupError) {
          t.addSuppressed(cleanupError);
        }
      }
      try {
        checkCuVSError(cuvsCagraIndexDestroy(index), "cuvsCagraIndexDestroy");
      } catch (Throwable cleanupError) {
        t.addSuppressed(cleanupError);
      }
      throw t;
    } finally {
      Files.deleteIfExists(tmpIndexFile);
    }
  }

  private static final class DatasetCloseDelegate implements AutoCloseable {
    private MemorySegment handle;

    private DatasetCloseDelegate(MemorySegment handle) {
      this.handle = handle;
    }

    @Override
    public void close() {
      if (handle != null && handle.address() != 0) {
        checkCuVSError(cuvsDatasetDestroy(handle), "cuvsDatasetDestroy");
        handle = MemorySegment.NULL;
      }
    }
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
   * Merges multiple CAGRA indexes into a single index, keeping only the rows selected by
   * {@code rowFilter}. See {@link CagraIndex#merge(CagraIndex[], CagraIndexParams, BitSet)} for the
   * meaning of the filter.
   *
   * @param indexes     Array of CAGRA indexes to merge
   * @param mergeParams Parameters to control the merge operation, or null to use defaults
   * @param rowFilter   The rows to keep, or null to keep all of them. A BitSet shorter than the
   *                    total row count is valid: the rows beyond its logical length are treated as
   *                    clear (dropped). See {@link CagraIndex#merge(CagraIndex[], CagraIndexParams,
   *                    BitSet)} for full semantics.
   * @return A new merged CAGRA index
   */
  public static CagraIndex merge(
      CagraIndex[] indexes, CagraIndexParams mergeParams, BitSet rowFilter) {
    if (indexes == null || indexes.length == 0) {
      throw new IllegalArgumentException("At least one index must be provided for merging");
    }
    CuVSResources resources = indexes[0].getCuVSResources();
    for (int i = 1; i < indexes.length; i++) {
      if (!resources.equals(indexes[i].getCuVSResources())) {
        throw new IllegalArgumentException("All indexes must use the same CuVSResources instance");
      }
    }

    try (var localArena = Arena.ofConfined()) {
      MemorySegment indexesSegment =
          localArena.allocate(indexes.length * ValueLayout.ADDRESS.byteSize());

      long mergedRowCount = 0;
      for (int i = 0; i < indexes.length; i++) {
        CagraIndexImpl indexImpl = (CagraIndexImpl) indexes[i];
        indexesSegment.setAtIndex(
            ValueLayout.ADDRESS, i, indexImpl.cagraIndexReference.getMemorySegment());
        if (rowFilter != null) {
          mergedRowCount += indexImpl.size();
        }
      }
      if (rowFilter != null) {
        if (rowFilter.length() > mergedRowCount) {
          throw new IllegalArgumentException(
              "rowFilter selects row "
                  + (rowFilter.length() - 1)
                  + " but the indexes only hold "
                  + mergedRowCount
                  + " rows");
        }
        if (rowFilter.isEmpty()) {
          throw new IllegalArgumentException("rowFilter keeps no rows, there is nothing to merge");
        }
      }

      var mergedIndex = createCagraIndex();
      CagraIndexImpl merged = null;
      try (var nativeMergeParams = segmentFromIndexParams(mergeParams);
          var resourcesAccessor = resources.access()) {
        var cuvsRes = resourcesAccessor.handle();

        // The words the merge filter points at have to outlive the merge call, so the
        // allocation is held open around it rather than inside the helper that fills
        // the filter in.
        MemorySegment mergeFilter = cuvsFilter.allocate(localArena);
        try (@SuppressWarnings("unused")
            var filterWords =
                allocateRowFilter(cuvsRes, localArena, mergeFilter, rowFilter, mergedRowCount)) {
          MemorySegment mergedDatasetPtr = localArena.allocate(cuvsDataset_t);
          checkCuVSError(cuvsDatasetCreate(mergedDatasetPtr), "cuvsDatasetCreate");
          MemorySegment mergedDataset = mergedDatasetPtr.get(cuvsDataset_t, 0);
          AutoCloseable datasetOwner = new DatasetCloseDelegate(mergedDataset);
          try {
            checkCuVSError(
                cuvsCagraMerge(
                    cuvsRes,
                    nativeMergeParams.handle(),
                    indexesSegment,
                    indexes.length,
                    mergeFilter,
                    mergedDataset,
                    mergedIndex),
                "cuvsCagraMerge");
            merged =
                new CagraIndexImpl(new IndexReference(mergedIndex, null, datasetOwner), resources);
            return merged;
          } catch (Throwable e) {
            try {
              datasetOwner.close();
            } catch (Exception closeError) {
              e.addSuppressed(closeError);
            }
            throw e;
          }
        }
      } catch (Throwable t) {
        try {
          if (merged != null) {
            // The merged index owns the dataset by now, so close it rather than only destroying
            // the handle.
            merged.close();
          } else {
            checkCuVSError(cuvsCagraIndexDestroy(mergedIndex), "cuvsCagraIndexDestroy");
          }
        } catch (Throwable cleanupError) {
          t.addSuppressed(cleanupError);
        }
        throw t;
      }
    }
  }

  /**
   * Fills {@code mergeFilter} in and returns the device allocation backing it, which the caller has
   * to keep open until the merge returns. A null {@code rowFilter} produces a NO_FILTER and an empty allocation.
   *
   * <p> cuvs reads the bitset as a vector of 32 bit words covering {@code mergedRowCount} rows, and derives the row
   * count of the merged index from the number of bits that are set, so the words have to cover every row rather
   * than stop at the last one that survives.
   */
  private static CloseableRMMAllocation allocateRowFilter(
      long cuvsRes, Arena arena, MemorySegment mergeFilter, BitSet rowFilter, long mergedRowCount) {
    if (rowFilter == null) {
      cuvsFilter.type(mergeFilter, NO_FILTER());
      cuvsFilter.addr(mergeFilter, 0);
      return CloseableRMMAllocation.EMPTY;
    }

    long words = (mergedRowCount + 31) / 32;
    long bytes = C_INT_BYTE_SIZE * words;
    MemorySegment hostWords =
        buildMemorySegment(arena, rowFilter.toLongArray(), (mergedRowCount + 63) / 64);

    var deviceWords = allocateRMMSegment(cuvsRes, bytes);
    try {
      Util.cudaMemcpyAsync(
          deviceWords.handle(), hostWords, bytes, HOST_TO_DEVICE, Util.getStream(cuvsRes));
      checkCuVSError(cuvsStreamSync(cuvsRes), "cuvsStreamSync");

      MemorySegment filterTensor =
          prepareTensor(arena, deviceWords.handle(), new long[] {words}, kDLUInt(), 32, kDLCUDA());
      cuvsFilter.type(mergeFilter, BITSET());
      cuvsFilter.addr(mergeFilter, filterTensor.address());
      return deviceWords;
    } catch (Throwable t) {
      try {
        deviceWords.close();
      } catch (Exception closeError) {
        t.addSuppressed(closeError);
      }
      throw t;
    }
  }

  /**
   * Builder helps configure and create an instance of {@link CagraIndex}.
   */
  public static class Builder implements CagraIndex.Builder {

    private CuVSMatrix dataset;
    private InputStream inputStream;
    private CagraIndex.DeserializeDataset outDataset;
    private CagraIndexParams cagraIndexParams;
    private final CuVSResources cuvsResources;
    private CuVSMatrix graph;

    public Builder(CuVSResources cuvsResources) {
      this.cuvsResources = cuvsResources;
    }

    @Override
    public Builder from(InputStream inputStream) {
      this.inputStream = inputStream;
      this.outDataset = null;
      return this;
    }

    @Override
    public Builder from(InputStream inputStream, CagraIndex.DeserializeDataset outDataset) {
      this.inputStream = inputStream;
      this.outDataset = Objects.requireNonNull(outDataset);
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
        return outDataset == null
            ? new CagraIndexImpl(inputStream, cuvsResources)
            : new CagraIndexImpl(inputStream, cuvsResources, outDataset);
      } else if (graph != null) {
        if (cagraIndexParams == null || dataset == null) {
          throw new IllegalArgumentException(
              "In order to reconstruct a CAGRA index from a graph, "
                  + "you must specify the original dataset and the metric used.");
        }
        return new CagraIndexImpl(
            cagraIndexParams.getCuvsDistanceType(), graph, dataset, cuvsResources);
      } else if (dataset != null) {
        return new CagraIndexImpl(cagraIndexParams, dataset, cuvsResources);
      } else {
        throw new IllegalArgumentException("dataset must be provided");
      }
    }
  }

  /**
   * Holds the memory reference to a CAGRA index.
   */
  public static class IndexReference {

    private final MemorySegment memorySegment;
    private final CuVSMatrix dataset;
    private final AutoCloseable datasetOwner;

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
      this(indexMemorySegment, dataset, dataset);
    }

    private IndexReference(
        MemorySegment indexMemorySegment, CuVSMatrix dataset, AutoCloseable datasetOwner) {
      this.memorySegment = indexMemorySegment;
      this.dataset = dataset;
      this.datasetOwner = datasetOwner;
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

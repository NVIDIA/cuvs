/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal;

import static com.nvidia.cuvs.internal.CuVSParamsHelper.createHnswAceParamsNative;
import static com.nvidia.cuvs.internal.CuVSParamsHelper.createHnswIndexParams;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_FLOAT;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_LONG;
import static com.nvidia.cuvs.internal.common.Util.buildMemorySegment;
import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.common.Util.prepareTensor;
import static com.nvidia.cuvs.internal.panama.headers_h.*;

import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CuVSHostMatrix;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.HnswAceParams;
import com.nvidia.cuvs.HnswIndex;
import com.nvidia.cuvs.HnswIndexParams;
import com.nvidia.cuvs.HnswQuery;
import com.nvidia.cuvs.HnswSearchParams;
import com.nvidia.cuvs.SearchResults;
import com.nvidia.cuvs.internal.common.CloseableHandle;
import com.nvidia.cuvs.internal.panama.DLDataType;
import com.nvidia.cuvs.internal.panama.cuvsHnswAceParams;
import com.nvidia.cuvs.internal.panama.cuvsHnswIndex;
import com.nvidia.cuvs.internal.panama.cuvsHnswIndexParams;
import com.nvidia.cuvs.internal.panama.cuvsHnswSearchParams;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;
import java.util.UUID;

/**
 * {@link HnswIndex} encapsulates a HNSW index, along with methods to interact
 * with it.
 *
 * @since 25.02
 */
public class HnswIndexImpl implements HnswIndex {

  private final CuVSResources resources;
  private final HnswIndexParams hnswIndexParams;
  private final IndexReference hnswIndexReference;
  private final Object lifecycleLock = new Object();
  private boolean destroyed;

  /**
   * Constructor for loading the index from an {@link InputStream}.
   *
   * NOTE: only float32 indexes can be loaded through this binding; the index file must have
   * been serialized from a float32 dataset.
   *
   * @param inputStream an instance of stream to read the index bytes from
   * @param resources   an instance of {@link CuVSResources}
   */
  private HnswIndexImpl(
      InputStream inputStream, CuVSResources resources, HnswIndexParams hnswIndexParams)
      throws Throwable {
    this.hnswIndexParams = hnswIndexParams;
    this.resources = resources;
    this.hnswIndexReference = deserialize(inputStream);
  }

  /**
   * Constructor for creating index from an existing IndexReference
   *
   * @param indexReference the index reference
   * @param resources      an instance of {@link CuVSResources}
   * @param hnswIndexParams an instance of {@link HnswIndexParams}
   */
  private HnswIndexImpl(
      IndexReference indexReference, CuVSResources resources, HnswIndexParams hnswIndexParams) {
    this.hnswIndexParams = hnswIndexParams;
    this.resources = resources;
    this.hnswIndexReference = indexReference;
  }

  /**
   * Invokes the native destroy_hnsw_index to de-allocate the HNSW index
   */
  @Override
  public void close() {
    synchronized (lifecycleLock) {
      checkNotDestroyed();
      try {
        int returnValue = cuvsHnswIndexDestroy(hnswIndexReference.getMemorySegment());
        checkCuVSError(returnValue, "cuvsHnswIndexDestroy");
      } finally {
        destroyed = true;
      }
    }
  }

  /** Must be called while holding {@link #lifecycleLock}. */
  private void checkNotDestroyed() {
    assert Thread.holdsLock(lifecycleLock);
    if (destroyed) {
      throw new IllegalStateException("destroyed");
    }
  }

  /**
   * Invokes the native search_hnsw_index via the Panama API for searching a HNSW
   * index.
   *
   * @param query an instance of {@link HnswQuery} holding the query vectors and
   *              other parameters
   * @return an instance of {@link HnswSearchResults} containing the results
   */
  @Override
  public SearchResults search(HnswQuery query) throws Throwable {
    synchronized (lifecycleLock) {
      checkNotDestroyed();
      return doSearch(query);
    }
  }

  private SearchResults doSearch(HnswQuery query) throws Throwable {
    try (var localArena = Arena.ofConfined()) {
      int topK = query.getTopK();
      float[][] queryVectors = query.getQueryVectors();
      int numQueries = queryVectors.length;
      long numBlocks = (long) topK * numQueries;
      int vectorDimension = numQueries > 0 ? queryVectors[0].length : 0;

      SequenceLayout neighborsSequenceLayout = MemoryLayout.sequenceLayout(numBlocks, C_LONG);
      SequenceLayout distancesSequenceLayout = MemoryLayout.sequenceLayout(numBlocks, C_FLOAT);
      // TODO: these could be CuVSHostMatrix
      MemorySegment neighborsMemorySegment = localArena.allocate(neighborsSequenceLayout);
      MemorySegment distancesMemorySegment = localArena.allocate(distancesSequenceLayout);
      MemorySegment querySeg = buildMemorySegment(localArena, queryVectors);

      long[] queriesShape = {numQueries, vectorDimension};
      MemorySegment queriesTensor =
          prepareTensor(localArena, querySeg, queriesShape, kDLFloat(), 32, kDLCPU());
      long[] neighborsShape = {numQueries, topK};
      // TODO: check type code and bits across all implementations -- they are inconsistent
      MemorySegment neighborsTensor =
          prepareTensor(
              localArena, neighborsMemorySegment, neighborsShape, kDLUInt(), 64, kDLCPU());
      long[] distancesShape = {numQueries, topK};
      MemorySegment distancesTensor =
          prepareTensor(
              localArena, distancesMemorySegment, distancesShape, kDLFloat(), 32, kDLCPU());

      try (var resourcesAccessor = query.getResources().access()) {
        var cuvsRes = resourcesAccessor.handle();
        int returnValue = cuvsStreamSync(cuvsRes);
        checkCuVSError(returnValue, "cuvsStreamSync");

        returnValue =
            cuvsHnswSearch(
                cuvsRes,
                segmentFromSearchParams(localArena, query.getHnswSearchParams()),
                hnswIndexReference.getMemorySegment(),
                queriesTensor,
                neighborsTensor,
                distancesTensor);
        checkCuVSError(returnValue, "cuvsHnswSearch");

        returnValue = cuvsStreamSync(cuvsRes);
        checkCuVSError(returnValue, "cuvsStreamSync");
      }

      return HnswSearchResults.create(
          neighborsSequenceLayout,
          distancesSequenceLayout,
          neighborsMemorySegment,
          distancesMemorySegment,
          topK,
          query.getMapping(),
          numQueries);
    }
  }

  private static IndexReference createHnswIndex() {
    try (var localArena = Arena.ofConfined()) {
      MemorySegment indexPtrPtr = localArena.allocate(cuvsHnswIndex_t);
      // cuvsHnswIndexCreate gets a pointer to a cuvsHnswIndex_t, which is defined as a pointer to
      // cuvsHnswIndex.
      // It's basically a "out" parameter: the C functions will create the index and "return back" a
      // pointer to it.
      // The "out parameter" pointer is needed only for the duration of the function invocation (it
      // could be a stack
      // pointer, in C) so we allocate it from our localArena.
      var returnValue = cuvsHnswIndexCreate(indexPtrPtr);
      checkCuVSError(returnValue, "cuvsHnswIndexCreate");
      return new IndexReference(indexPtrPtr.get(cuvsHnswIndex_t, 0));
    }
  }

  /**
   * Gets an instance of {@link IndexReference} by deserializing a HNSW index
   * using an {@link InputStream}.
   *
   * @param inputStream an instance of {@link InputStream}
   * @return an instance of {@link IndexReference}.
   */
  private IndexReference deserialize(InputStream inputStream) throws Throwable {
    if (hnswIndexParams.getVectorDimension() <= 0) {
      // A dim of 0 would silently construct a zero-dimensional native space whose searches
      // return garbage; the builder's default vectorDimension is 0, so require an explicit one.
      throw new IllegalArgumentException(
          "vectorDimension must be set to the positive dimension of the serialized index");
    }
    Path tmpIndexFile =
        Files.createTempFile(resources.tempDirectory(), UUID.randomUUID().toString(), ".hnsw")
            .toAbsolutePath();
    IndexReference indexReference = null;

    try {
      try (inputStream; var outputStream = Files.newOutputStream(tmpIndexFile)) {
        inputStream.transferTo(outputStream);
      }

      try (var localArena = Arena.ofConfined()) {
        MemorySegment pathSeg = buildMemorySegment(localArena, tmpIndexFile.toString());
        indexReference = createHnswIndex();

        // Deserialization supports float32 indexes only: passing a null dataset applies the
        // float32 default, and HnswQuery likewise only issues float32 queries.
        initializeIndexDType(indexReference.memorySegment, null);

        try (var params = segmentFromIndexParams(hnswIndexParams);
            var cuvsResourcesAccessor = resources.access()) {
          checkCuVSError(
              cuvsHnswDeserialize(
                  cuvsResourcesAccessor.handle(),
                  params.handle(),
                  pathSeg,
                  hnswIndexParams.getVectorDimension(),
                  hnswIndexParams.getMetric().value,
                  indexReference.memorySegment),
              "cuvsHnswDeserialize");
        }
      }

      return indexReference;
    } catch (Throwable failure) {
      if (indexReference != null) {
        destroyIndexAfterFailure(indexReference.memorySegment, failure);
      }
      throw failure;
    } finally {
      Files.deleteIfExists(tmpIndexFile);
    }
  }

  /** Allocates the configured index parameters in a native handle. */
  private static CloseableHandle segmentFromIndexParams(HnswIndexParams params) {
    var hnswParams = createHnswIndexParams();
    cuvsHnswIndexParams.hierarchy(hnswParams.handle(), params.getHierarchy().value);
    cuvsHnswIndexParams.ef_construction(hnswParams.handle(), params.getEfConstruction());
    cuvsHnswIndexParams.num_threads(hnswParams.handle(), params.getNumThreads());
    cuvsHnswIndexParams.M(hnswParams.handle(), params.getM());
    cuvsHnswIndexParams.metric(hnswParams.handle(), params.getMetric().value);
    return hnswParams;
  }

  /**
   * Allocates the configured search parameters in the MemorySegment.
   */
  private static MemorySegment segmentFromSearchParams(Arena arena, HnswSearchParams params) {
    MemorySegment seg = cuvsHnswSearchParams.allocate(arena);
    cuvsHnswSearchParams.ef(seg, params.ef());
    cuvsHnswSearchParams.num_threads(seg, params.numThreads());
    return seg;
  }

  public static HnswIndex.Builder newBuilder(CuVSResources cuvsResources) {
    return new HnswIndexImpl.Builder(Objects.requireNonNull(cuvsResources));
  }

  /**
   * Builds an HNSW index on the GPU.
   *
   * @param resources The CuVS resources
   * @param hnswParams Parameters for the HNSW index
   * @param dataset The dataset to build the index from
   * @return A new HNSW index ready for search
   * @throws Throwable if an error occurs during building
   */
  public static HnswIndex build(CuVSResources resources, HnswIndexParams hnswParams, CuVSMatrix dataset)
      throws Throwable {
    Objects.requireNonNull(resources);
    Objects.requireNonNull(hnswParams);
    Objects.requireNonNull(dataset);
    if (dataset.dataType() != CuVSMatrix.DataType.FLOAT) {
      // HnswQuery only carries float queries, so an index built from any other element type
      // could never be searched through this binding.
      throw new UnsupportedOperationException(
          "HnswIndex.build supports float32 datasets only; got " + dataset.dataType());
    }

    // Create HNSW index
    MemorySegment hnswIndex = createHnswIndexHandle();

    try {
      initializeIndexDType(hnswIndex, dataset);
      try (var localArena = Arena.ofConfined();
          var hnswParamsHandle = segmentFromIndexParams(hnswParams);
          var aceParamsHandle = createHnswAceParams(localArena, hnswParams.getAceParams())) {

        MemorySegment hnswParamsMemorySegment = hnswParamsHandle.handle();

        if (aceParamsHandle != CloseableHandle.NULL) {
          cuvsHnswIndexParams.ace_params(hnswParamsMemorySegment, aceParamsHandle.handle());
        }

        MemorySegment datasetTensor = prepareTensorFromMatrix(localArena, dataset);

        try (var resourcesAccessor = resources.access()) {
          var cuvsRes = resourcesAccessor.handle();

          int returnValue =
              cuvsHnswBuild(cuvsRes, hnswParamsMemorySegment, datasetTensor, hnswIndex);
          checkCuVSError(returnValue, "cuvsHnswBuild");

          returnValue = cuvsStreamSync(cuvsRes);
          checkCuVSError(returnValue, "cuvsStreamSync");
        }
      }
      return new HnswIndexImpl(new IndexReference(hnswIndex), resources, hnswParams);
    } catch (Throwable failure) {
      destroyIndexAfterFailure(hnswIndex, failure);
      throw failure;
    }
  }

  private static void destroyIndexAfterFailure(MemorySegment hnswIndex, Throwable failure) {
    try {
      checkCuVSError(cuvsHnswIndexDestroy(hnswIndex), "cuvsHnswIndexDestroy");
    } catch (Throwable cleanupFailure) {
      failure.addSuppressed(cleanupFailure);
    }
  }

  private static CloseableHandle createHnswAceParams(Arena arena, HnswAceParams aceParams) {
    if (aceParams == null) {
      return CloseableHandle.NULL;
    }

    var params = createHnswAceParamsNative();
    MemorySegment seg = params.handle();

    cuvsHnswAceParams.npartitions(seg, aceParams.getNpartitions());
    cuvsHnswAceParams.use_disk(seg, aceParams.isUseDisk());
    cuvsHnswAceParams.max_host_memory_gb(seg, aceParams.getMaxHostMemoryGb());
    cuvsHnswAceParams.max_gpu_memory_gb(seg, aceParams.getMaxGpuMemoryGb());

    String buildDir = aceParams.getBuildDir();
    if (buildDir != null) {
      MemorySegment buildDirSeg = arena.allocateFrom(buildDir);
      cuvsHnswAceParams.build_dir(seg, buildDirSeg);
    }

    return params;
  }

  private static MemorySegment prepareTensorFromMatrix(Arena arena, CuVSMatrix dataset) {
    if (dataset instanceof CuVSMatrixInternal matrixInternal) {
      return prepareTensor(
          arena,
          matrixInternal.memorySegment(),
          new long[]{dataset.size(), dataset.columns()},
          matrixInternal.code(),
          matrixInternal.bits(),
          kDLCPU());
    }
    throw new IllegalArgumentException("Unsupported matrix type for build");
  }

  /**
   * Creates an HNSW index from an existing CAGRA index.
   *
   * @param hnswParams Parameters for the HNSW index
   * @param cagraIndex The CAGRA index to convert from
   * @return A new HNSW index for in-memory indexes, or null for disk-based indexes
   * @throws Throwable if an error occurs during conversion
   */
  public static HnswIndex fromCagra(HnswIndexParams hnswParams, CagraIndex cagraIndex)
      throws Throwable {
    Objects.requireNonNull(hnswParams);
    Objects.requireNonNull(cagraIndex);

    // Get the CAGRA index implementation to access internals
    if (!(cagraIndex instanceof CagraIndexImpl)) {
      throw new IllegalArgumentException("Invalid CagraIndex implementation");
    }
    CagraIndexImpl cagraImpl = (CagraIndexImpl) cagraIndex;
    CuVSResources resources = cagraImpl.getCuVSResources();

    // Create HNSW index
    MemorySegment hnswIndex = createHnswIndexHandle();

    try {
      CuVSMatrix dataset = cagraImpl.getDatasetForConversion();
      initializeIndexDType(hnswIndex, dataset);

      try (var localArena = Arena.ofConfined();
          var hnswParamsHandle = segmentFromIndexParams(hnswParams)) {
        MemorySegment hnswParamsMemorySegment = hnswParamsHandle.handle();

        try (var resourcesAccessor = resources.access()) {
          var cuvsRes = resourcesAccessor.handle();

          int returnValue;
          if (dataset instanceof CuVSHostMatrix) {
            // Hand the host dataset to the conversion so the native side does not have to
            // copy the full dataset back out of the CAGRA index.
            MemorySegment datasetTensor = prepareTensorFromMatrix(localArena, dataset);
            returnValue =
                cuvsHnswFromCagraWithDataset(
                    cuvsRes,
                    hnswParamsMemorySegment,
                    cagraImpl.getCagraIndexReference(),
                    hnswIndex,
                    datasetTensor);
            checkCuVSError(returnValue, "cuvsHnswFromCagraWithDataset");
          } else {
            returnValue =
                cuvsHnswFromCagra(
                    cuvsRes, hnswParamsMemorySegment, cagraImpl.getCagraIndexReference(), hnswIndex);
            checkCuVSError(returnValue, "cuvsHnswFromCagra");
          }

          returnValue = cuvsStreamSync(cuvsRes);
          checkCuVSError(returnValue, "cuvsStreamSync");
        }
      }
      return new HnswIndexImpl(new IndexReference(hnswIndex), resources, hnswParams);
    } catch (Throwable failure) {
      destroyIndexAfterFailure(hnswIndex, failure);
      throw failure;
    }
  }

  /**
   * Creates a new HNSW index handle.
   */
  private static MemorySegment createHnswIndexHandle() {
    try (var localArena = Arena.ofConfined()) {
      MemorySegment indexPtrPtr = localArena.allocate(cuvsHnswIndex_t);
      var returnValue = cuvsHnswIndexCreate(indexPtrPtr);
      checkCuVSError(returnValue, "cuvsHnswIndexCreate");
      return indexPtrPtr.get(cuvsHnswIndex_t, 0);
    }
  }

  private static void initializeIndexDType(MemorySegment hnswIndex, CuVSMatrix dataset) {
    int bits = 32;
    int code = kDLFloat();

    if (dataset instanceof CuVSMatrixInternal matrixInternal) {
      bits = matrixInternal.bits();
      code = matrixInternal.code();
    } else if (dataset != null) {
      bits = bitsFromDataType(dataset.dataType());
      code = CuVSMatrixInternal.code(dataset.dataType());
    }

    try (var localArena = Arena.ofConfined()) {
      MemorySegment dtype = DLDataType.allocate(localArena);
      DLDataType.bits(dtype, (byte) bits);
      DLDataType.code(dtype, (byte) code);
      DLDataType.lanes(dtype, (byte) 1);
      cuvsHnswIndex.dtype(hnswIndex, dtype);
    }
  }

  private static int bitsFromDataType(CuVSMatrix.DataType dataType) {
    return switch (dataType) {
      case BYTE -> 8;
      default -> 32;
    };
  }

  /**
   * Builder helps configure and create an instance of {@link HnswIndex}.
   */
  public static class Builder implements HnswIndex.Builder {

    private final CuVSResources cuvsResources;
    private InputStream inputStream;
    private HnswIndexParams hnswIndexParams;

    /**
     * Constructs this Builder with an instance of {@link CuVSResources}.
     *
     * @param cuvsResources an instance of {@link CuVSResources}
     */
    public Builder(CuVSResources cuvsResources) {
      this.cuvsResources = cuvsResources;
    }

    /**
     * Sets an instance of InputStream typically used when index deserialization is
     * needed.
     *
     * @param inputStream an instance of {@link InputStream}
     * @return an instance of this Builder
     */
    @Override
    public Builder from(InputStream inputStream) {
      this.inputStream = inputStream;
      return this;
    }

    /**
     * Registers an instance of configured {@link HnswIndexParams} with this
     * Builder.
     *
     * @param hnswIndexParameters An instance of HnswIndexParams.
     * @return An instance of this Builder.
     */
    @Override
    public Builder withIndexParams(HnswIndexParams hnswIndexParameters) {
      this.hnswIndexParams = hnswIndexParameters;
      return this;
    }

    /**
     * Builds and returns an instance of CagraIndex.
     *
     * @return an instance of CagraIndex
     */
    @Override
    public HnswIndexImpl build() throws Throwable {
      return new HnswIndexImpl(
          Objects.requireNonNull(inputStream, "input stream must be set before build"),
          cuvsResources,
          Objects.requireNonNull(hnswIndexParams, "index parameters must be set before build"));
    }
  }

  /**
   * Holds the memory reference to a HNSW index.
   */
  protected static class IndexReference {

    private final MemorySegment memorySegment;

    /**
     * Constructs CagraIndexReference with an instance of MemorySegment passed as a
     * parameter.
     *
     * @param indexMemorySegment the MemorySegment instance to use for containing
     *                           index reference
     */
    protected IndexReference(MemorySegment indexMemorySegment) {
      this.memorySegment = indexMemorySegment;
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

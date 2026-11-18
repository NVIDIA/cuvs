/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal;

import static com.nvidia.cuvs.internal.CuVSParamsHelper.createVamanaIndexParams;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_INT;
import static com.nvidia.cuvs.internal.common.Util.buildMemorySegment;
import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.common.Util.prepareTensor;
import static com.nvidia.cuvs.internal.panama.headers_h.*;

import com.nvidia.cuvs.CuVSHostMatrix;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.VamanaIndex;
import com.nvidia.cuvs.VamanaIndexParams;
import com.nvidia.cuvs.internal.common.CloseableHandle;
import com.nvidia.cuvs.internal.panama.cuvsVamanaIndexParams;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Objects;

/**
 * {@link VamanaIndex} encapsulates a Vamana index, along with methods to build
 * it on the GPU and serialize it in the DiskANN file format.
 * <p>
 * cuVS provides build and serialize for Vamana but no search entry point, so
 * this class deliberately exposes no search method.
 *
 * @since 26.12
 */
public class VamanaIndexImpl implements VamanaIndex {

  private final CuVSResources resources;
  private final MemorySegment vamanaIndexReference;
  private final CuVSMatrix dataset;
  private final boolean ownsDataset;
  private boolean destroyed;

  private VamanaIndexImpl(
      VamanaIndexParams indexParameters,
      CuVSMatrix dataset,
      boolean ownsDataset,
      CuVSResources resources) {
    Objects.requireNonNull(dataset);
    this.resources = resources;
    // the native index may retain a non-owning device view of the dataset, so
    // we hold a reference to keep it alive for at least as long as the index
    this.dataset = dataset;
    this.ownsDataset = ownsDataset;
    if (!(dataset instanceof CuVSMatrixInternal internalDataset)) {
      throw new IllegalArgumentException(
          "dataset must be created through CuVSMatrix, was " + dataset.getClass().getName());
    }
    checkSupportedDataType(dataset.dataType());
    this.vamanaIndexReference = build(indexParameters, internalDataset);
  }

  /**
   * The native Vamana builder is instantiated for {@code float}, {@code half},
   * {@code int8}, and {@code uint8} only. Of those, {@code int8} has no
   * corresponding {@link CuVSMatrix.DataType}. Reject anything else here rather
   * than inside a kernel.
   */
  private static void checkSupportedDataType(CuVSMatrix.DataType dataType) {
    switch (dataType) {
      case FLOAT, HALF, BYTE -> {}
      default ->
          throw new IllegalArgumentException(
              "Vamana supports FLOAT, HALF, and BYTE datasets, was " + dataType);
    }
  }

  private void checkNotDestroyed() {
    if (destroyed) {
      throw new IllegalStateException("destroyed");
    }
  }

  @Override
  public void close() {
    checkNotDestroyed();
    destroyed = true;
    RuntimeException failure = null;
    try {
      checkCuVSError(cuvsVamanaIndexDestroy(vamanaIndexReference), "cuvsVamanaIndexDestroy");
    } catch (RuntimeException e) {
      failure = e;
    }
    if (ownsDataset) {
      // attempt this even if the index failed to destroy, so a dataset this
      // index created is never stranded
      try {
        dataset.close();
      } catch (Exception e) {
        RuntimeException wrapped = new RuntimeException("Failed to close the Vamana dataset", e);
        if (failure == null) {
          failure = wrapped;
        } else {
          failure.addSuppressed(wrapped);
        }
      }
    }
    if (failure != null) {
      throw failure;
    }
  }

  /**
   * Creates the native index handle. The handle is a native heap allocation and
   * must be released with {@code cuvsVamanaIndexDestroy}, so it is deliberately
   * not tied to an {@link Arena}.
   */
  private static MemorySegment createVamanaIndex() {
    try (var localArena = Arena.ofConfined()) {
      MemorySegment indexPtrPtr = localArena.allocate(cuvsVamanaIndex_t);
      checkCuVSError(cuvsVamanaIndexCreate(indexPtrPtr), "cuvsVamanaIndexCreate");
      return indexPtrPtr.get(cuvsVamanaIndex_t, 0);
    }
  }

  /**
   * Populates a native parameter struct from the Java parameters. A null
   * argument leaves the native defaults in place.
   */
  private static CloseableHandle segmentFromIndexParams(VamanaIndexParams params) {
    var handle = createVamanaIndexParams();
    if (params == null) {
      return handle;
    }
    try {
      MemorySegment seg = handle.handle();
      cuvsVamanaIndexParams.graph_degree(seg, params.getGraphDegree());
      cuvsVamanaIndexParams.visited_size(seg, params.getVisitedSize());
      cuvsVamanaIndexParams.vamana_iters(seg, params.getVamanaIters());
      cuvsVamanaIndexParams.alpha(seg, params.getAlpha());
      cuvsVamanaIndexParams.max_fraction(seg, params.getMaxFraction());
      cuvsVamanaIndexParams.batch_base(seg, params.getBatchBase());
      cuvsVamanaIndexParams.queue_size(seg, params.getQueueSize());
      cuvsVamanaIndexParams.reverse_batchsize(seg, params.getReverseBatchSize());
      cuvsVamanaIndexParams.metric(seg, params.getMetric().value);
      return handle;
    } catch (RuntimeException | Error e) {
      handle.close();
      throw e;
    }
  }

  /**
   * The initial graph as a rank-2 host tensor.
   *
   * <p>Built without strides rather than through {@code toTensor}, which emits
   * {@code {rowStride, columnStride}} whenever a row stride is set and so would pass a column
   * stride of -1, which the native conversion rejects. A graph with padded rows is rejected here
   * instead, since the native build needs contiguous rows.
   */
  private static MemorySegment initialGraphTensor(CuVSHostMatrix graph, Arena arena) {
    var internal = (CuVSMatrixInternal) graph;
    long rowStride = internal.rowStride();
    if (rowStride > 0 && rowStride != graph.columns()) {
      throw new IllegalArgumentException(
          "initialGraph must be contiguous, but its row stride ("
              + rowStride
              + ") does not match its column count ("
              + graph.columns()
              + ")");
    }
    return prepareTensor(
        arena,
        internal.memorySegment(),
        new long[] {graph.size(), graph.columns()},
        internal.code(),
        internal.bits(),
        kDLCPU());
  }

  /**
   * Invokes the native {@code cuvsVamanaBuild} function to build the
   * {@link VamanaIndex}.
   *
   * @return the handle of the built index
   */
  private MemorySegment build(VamanaIndexParams indexParameters, CuVSMatrixInternal dataset) {
    try (var indexParams = segmentFromIndexParams(indexParameters);
        var localArena = Arena.ofConfined()) {

      var datasetTensor = dataset.toTensor(localArena);
      var initialGraph = indexParameters == null ? null : indexParameters.getInitialGraph();
      if (initialGraph != null) {
        cuvsVamanaIndexParams.initial_graph(
            indexParams.handle(), initialGraphTensor(initialGraph, localArena));
      }
      var index = createVamanaIndex();
      try {
        try (var resourcesAccessor = resources.access()) {
          var cuvsRes = resourcesAccessor.handle();

          checkCuVSError(cuvsStreamSync(cuvsRes), "cuvsStreamSync");
          checkCuVSError(
              cuvsVamanaBuild(cuvsRes, indexParams.handle(), datasetTensor, index),
              "cuvsVamanaBuild");
          checkCuVSError(cuvsStreamSync(cuvsRes), "cuvsStreamSync");
        }
      } catch (RuntimeException | Error e) {
        // the index handle is a native allocation, so release it if the build
        // never completed
        checkCuVSError(cuvsVamanaIndexDestroy(index), "cuvsVamanaIndexDestroy");
        throw e;
      }
      return index;
    }
  }

  @Override
  public int getDimensions() {
    checkNotDestroyed();
    try (var localArena = Arena.ofConfined()) {
      MemorySegment dims = localArena.allocate(C_INT);
      checkCuVSError(cuvsVamanaIndexGetDims(vamanaIndexReference, dims), "cuvsVamanaIndexGetDims");
      return dims.get(C_INT, 0);
    }
  }

  @Override
  public void serialize(Path filePrefix, boolean includeDataset) {
    checkNotDestroyed();
    Objects.requireNonNull(filePrefix);
    try (var localArena = Arena.ofConfined();
        var resourcesAccessor = resources.access()) {
      MemorySegment prefix = buildMemorySegment(localArena, filePrefix.toAbsolutePath().toString());
      checkCuVSError(
          cuvsVamanaSerialize(
              resourcesAccessor.handle(), prefix, vamanaIndexReference, includeDataset),
          "cuvsVamanaSerialize");
    }
  }

  @Override
  public CuVSResources getCuVSResources() {
    return resources;
  }

  public static VamanaIndex.Builder newBuilder(CuVSResources cuvsResources) {
    return new Builder(Objects.requireNonNull(cuvsResources));
  }

  /**
   * Builder helps configure and create an instance of {@link VamanaIndex}.
   */
  public static class Builder implements VamanaIndex.Builder {

    private final CuVSResources cuvsResources;
    private CuVSMatrix dataset;
    private boolean ownsDataset;
    private VamanaIndexParams vamanaIndexParams;

    public Builder(CuVSResources cuvsResources) {
      this.cuvsResources = cuvsResources;
    }

    @Override
    public Builder withDataset(float[][] vectors) {
      // build the matrix first, then release any matrix this builder previously
      // created, so a second call cannot strand the first one
      CuVSMatrix created = CuVSMatrix.ofArray(vectors);
      releaseOwnedDataset();
      this.dataset = created;
      // we created it, so we close it
      this.ownsDataset = true;
      return this;
    }

    @Override
    public Builder withDataset(CuVSMatrix dataset) {
      releaseOwnedDataset();
      this.dataset = dataset;
      // the caller created it, so the caller closes it
      this.ownsDataset = false;
      return this;
    }

    private void releaseOwnedDataset() {
      if (ownsDataset && dataset != null) {
        try {
          dataset.close();
        } catch (Exception e) {
          throw new RuntimeException("Failed to close the previously supplied dataset", e);
        }
      }
      this.dataset = null;
      this.ownsDataset = false;
    }

    @Override
    public Builder withIndexParams(VamanaIndexParams vamanaIndexParameters) {
      this.vamanaIndexParams = vamanaIndexParameters;
      return this;
    }

    @Override
    public VamanaIndexImpl build() {
      if (dataset == null) {
        throw new IllegalArgumentException("dataset must be provided");
      }
      boolean transferred = false;
      try {
        VamanaIndexImpl index =
            new VamanaIndexImpl(vamanaIndexParams, dataset, ownsDataset, cuvsResources);
        // ownership now belongs to the index
        transferred = true;
        this.dataset = null;
        this.ownsDataset = false;
        return index;
      } finally {
        // a failed construction must not strand a matrix this builder created
        if (!transferred) {
          releaseOwnedDataset();
        }
      }
    }
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal;

import static com.nvidia.cuvs.internal.common.CloseableRMMAllocation.allocateRMMSegment;
import static com.nvidia.cuvs.internal.common.Util.CudaMemcpyKind.DEVICE_TO_HOST;
import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.common.Util.cudaMemcpyAsync;
import static com.nvidia.cuvs.internal.common.Util.getStream;
import static com.nvidia.cuvs.internal.common.Util.prepareTensor;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsCagraSearchMultiPartition;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsStreamSync;
import static com.nvidia.cuvs.internal.panama.headers_h.kDLCUDA;
import static com.nvidia.cuvs.internal.panama.headers_h.kDLFloat;
import static com.nvidia.cuvs.internal.panama.headers_h.kDLUInt;

import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraQuery;
import com.nvidia.cuvs.CagraSearchParams;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.FilterBitsetHandle;
import com.nvidia.cuvs.MultiPartitionSearchResults;
import com.nvidia.cuvs.internal.panama.cuvsFilter;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;

/**
 * JDK/Panama implementation of multi-partition CAGRA search. The public entry point
 * {@code com.nvidia.cuvs.MultiPartitionCagraSearch} delegates here via {@code CuVSProvider}.
 *
 * <h3>Algorithm (executed natively)</h3>
 * <ol>
 *   <li>For each (query, partition) pair, run CAGRA search into an internal
 *       [num_partitions, n_queries, k] device buffer.</li>
 *   <li>Apply per-partition distance post-processing on the intermediate buffer.</li>
 *   <li>Run a batched {@code raft::matrix::select_k} to pick the global top-k per query.</li>
 *   <li>Decode the select_k positions into {@code partition_ids} and {@code neighbors} outputs.</li>
 * </ol>
 */
public final class MultiPartitionCagraSearchImpl {

  private MultiPartitionCagraSearchImpl() {}

  /**
   * Acquires a reference to every non-null handle in {@code filters} for the duration of a search;
   * the returned resource releases them all on {@code close()}. Returns a no-op resource when there
   * are no filters. Rolls back already-acquired references if any handle has already been released.
   *
   * @throws IllegalStateException if a handle has already been fully released
   */
  private static AutoCloseable acquireFilterRefs(List<FilterBitsetHandle> filters) {
    if (filters == null) return () -> {};
    List<FilterBitsetHandle> acquired = new ArrayList<>(filters.size());
    try {
      for (FilterBitsetHandle f : filters) {
        if (f == null) continue;
        if (!f.tryIncRef()) {
          throw new IllegalStateException(
              "FilterBitsetHandle has already been released and cannot be used for search");
        }
        acquired.add(f);
      }
    } catch (RuntimeException e) {
      for (FilterBitsetHandle f : acquired) f.decRef();
      throw e;
    }
    return () -> {
      for (FilterBitsetHandle f : acquired) f.decRef();
    };
  }

  public static MultiPartitionSearchResults search(
      CuVSResources resources,
      List<CagraIndex> indices,
      CagraQuery query,
      int k,
      List<FilterBitsetHandle> filters)
      throws Throwable {
    int numPartitions = indices.size();
    if (numPartitions == 0) {
      return new MultiPartitionSearchResults(0, new int[0], new int[0], new float[0]);
    }

    CagraIndexImpl[] buffered = new CagraIndexImpl[numPartitions];
    for (int i = 0; i < numPartitions; i++) {
      CagraIndex idx = indices.get(i);
      if (!(idx instanceof CagraIndexImpl)) {
        throw new IllegalArgumentException(
            "Index at position " + i + " does not support buffered search");
      }
      buffered[i] = (CagraIndexImpl) idx;
    }

    // null/empty filters == fully unfiltered; otherwise one entry per partition (a null entry means
    // no filter for that partition). Validate the size and handle types up front.
    boolean hasFilters = filters != null && !filters.isEmpty();
    if (hasFilters && filters.size() != numPartitions) {
      throw new IllegalArgumentException(
          "filters must be null/empty (unfiltered) or have one entry per partition ("
              + numPartitions
              + "); got "
              + filters.size());
    }
    if (hasFilters) {
      for (int i = 0; i < numPartitions; i++) {
        FilterBitsetHandle f = filters.get(i);
        if (f != null && !(f instanceof FilterBitsetHandleImpl)) {
          throw new IllegalArgumentException(
              "filter for partition "
                  + i
                  + " must be a FilterBitsetHandle created via FilterBitsetHandle.create(...); got "
                  + f.getClass().getName());
        }
      }
    }

    var queryVectors = (CuVSMatrixInternal) query.getQueryVectors();
    int nQueries = (int) queryVectors.size();

    long partitionIdsBytes = (long) nQueries * k * Integer.BYTES; // uint32
    long neighborsBytes = (long) nQueries * k * Integer.BYTES; // uint32
    long distancesBytes = (long) nQueries * k * Float.BYTES;

    CagraSearchParams searchParameters = query.getCagraSearchParameters();

    // Hold a reference to the filter for the whole device operation so a concurrent close() (e.g.
    // eviction from a host-level cache) cannot free its device allocation while it is still in use.
    // Released after the resources block, once the search and its stream sync have completed.
    try (var filterRefs = acquireFilterRefs(filters);
        var resourcesAccessor = resources.access()) {
      long cuvsRes = resourcesAccessor.handle();
      var cuvsStream = getStream(cuvsRes);

      try (var partitionIdsDP = allocateRMMSegment(cuvsRes, partitionIdsBytes);
          var neighborsDP = allocateRMMSegment(cuvsRes, neighborsBytes);
          var distancesDP = allocateRMMSegment(cuvsRes, distancesBytes)) {

        // Upload host queries to device (the native call needs a device tensor). toDevice is a
        // cheap delegate wrapper when the matrix is already on device, so device callers pay
        // nothing; for host matrices it builds an owned device copy that this block closes.
        try (var arena = Arena.ofConfined();
            var deviceQueryVectors = (CuVSMatrixInternal) queryVectors.toDevice(resources)) {
          MemorySegment sp = CuVSParamsHelper.buildCagraSearchParams(arena, searchParameters);

          MemorySegment indexArray = arena.allocate(ValueLayout.ADDRESS, numPartitions);
          for (int i = 0; i < numPartitions; i++) {
            indexArray.setAtIndex(ValueLayout.ADDRESS, i, buffered[i].getIndexHandle());
          }

          MemorySegment queriesTensor = deviceQueryVectors.toTensor(arena);

          long[] outShape = {nQueries, k};
          MemorySegment partitionIdsTensor =
              prepareTensor(arena, partitionIdsDP.handle(), outShape, kDLUInt(), 32, kDLCUDA());
          MemorySegment neighborsTensor =
              prepareTensor(arena, neighborsDP.handle(), outShape, kDLUInt(), 32, kDLCUDA());
          MemorySegment distancesTensor =
              prepareTensor(arena, distancesDP.handle(), outShape, kDLFloat(), 32, kDLCUDA());

          // Per-partition filters: NULL when unfiltered, else an array of one cuvsFilter per
          // partition (NO_FILTER for a null entry, BITSET pointing at that partition's own bitset).
          MemorySegment filtersArg = MemorySegment.NULL;
          if (hasFilters) {
            filtersArg = cuvsFilter.allocateArray(numPartitions, arena);
            for (int i = 0; i < numPartitions; i++) {
              MemorySegment filterSeg = cuvsFilter.asSlice(filtersArg, i);
              FilterBitsetHandle f = filters.get(i);
              if (f != null) {
                FilterBitsetHandleImpl.DeviceData dev = ((FilterBitsetHandleImpl) f).getOrUpload();
                buildCuvsFilterStruct(arena, filterSeg, dev.combinedBitsetDP.handle(), dev.combinedWords);
              } else {
                cuvsFilter.type(filterSeg, 0 /* NO_FILTER */);
                cuvsFilter.addr(filterSeg, 0L);
              }
            }
          }

          checkCuVSError(
              cuvsCagraSearchMultiPartition(
                  cuvsRes,
                  sp,
                  numPartitions,
                  indexArray,
                  queriesTensor,
                  partitionIdsTensor,
                  neighborsTensor,
                  distancesTensor,
                  filtersArg),
              "cuvsCagraSearchMultiPartition");
        }

        // Copy the three small output arrays to host in a single allocation.
        try (var hostArena = Arena.ofConfined()) {
          MemorySegment hostBuf =
              hostArena.allocate(partitionIdsBytes + neighborsBytes + distancesBytes, Long.BYTES);
          MemorySegment hostPartitionIds = hostBuf.asSlice(0, partitionIdsBytes);
          MemorySegment hostNeighbors = hostBuf.asSlice(partitionIdsBytes, neighborsBytes);
          MemorySegment hostDistances =
              hostBuf.asSlice(partitionIdsBytes + neighborsBytes, distancesBytes);

          cudaMemcpyAsync(
              hostPartitionIds,
              partitionIdsDP.handle(),
              partitionIdsBytes,
              DEVICE_TO_HOST,
              cuvsStream);
          cudaMemcpyAsync(
              hostNeighbors, neighborsDP.handle(), neighborsBytes, DEVICE_TO_HOST, cuvsStream);
          cudaMemcpyAsync(
              hostDistances, distancesDP.handle(), distancesBytes, DEVICE_TO_HOST, cuvsStream);

          checkCuVSError(cuvsStreamSync(cuvsRes), "cuvsStreamSync after D2H copy");

          int total = nQueries * k;
          int[] partitionIds = new int[total];
          int[] selectedNeighbors = new int[total];
          float[] selectedDistances = new float[total];
          int count = 0;
          for (int j = 0; j < total; j++) {
            float distance = hostDistances.getAtIndex(ValueLayout.JAVA_FLOAT, j);
            // Unfilled top-k slots (fewer than k candidates passed the filter) carry a sentinel
            // distance of FLT_MAX. The sentinel neighbor index is not uniform across CAGRA search
            // algorithms (single-CTA clears the top bit and emits 0x7FFFFFFF, multi-CTA emits
            // 0xFFFFFFFF), so the distance is the reliable, algorithm-independent signal.
            if (distance == Float.MAX_VALUE) continue;

            int neighbor = hostNeighbors.getAtIndex(ValueLayout.JAVA_INT, j);
            if (neighbor < 0) {
              // A valid ordinal >= 2^31 does not fit in a signed int.
              throw new ArithmeticException(
                  "ordinal "
                      + Integer.toUnsignedLong(neighbor)
                      + " exceeds Integer.MAX_VALUE; partitions larger than 2^31 vectors are not"
                      + " supported by this API");
            }
            partitionIds[count] = hostPartitionIds.getAtIndex(ValueLayout.JAVA_INT, j);
            selectedNeighbors[count] = neighbor;
            selectedDistances[count] = distance;
            count++;
          }

          return new MultiPartitionSearchResults(
              count, partitionIds, selectedNeighbors, selectedDistances);
        }
      }
    }
  }

  /**
   * Populates one {@code cuvsFilter} struct for a single partition: a plain {@code BITSET} filter
   * whose address is a DLPack tensor over that partition's own device bitset.
   */
  private static void buildCuvsFilterStruct(
      Arena arena, MemorySegment filterSeg, MemorySegment bitsetHandle, long words) {
    long[] bitsetShape = {words};
    MemorySegment bitsetTensor =
        prepareTensor(arena, bitsetHandle, bitsetShape, kDLUInt(), 32, kDLCUDA());

    cuvsFilter.type(filterSeg, 1 /* BITSET */);
    cuvsFilter.addr(filterSeg, bitsetTensor.address());
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import com.nvidia.cuvs.spi.CuVSProvider;
import java.util.List;

/**
 * Performs an approximate nearest neighbor search across multiple CAGRA index partitions in a
 * single native call. The caller supplies one {@link CagraQuery} whose query matrix is searched
 * against every partition; cuVS performs the per-partition searches, the cross-partition top-k
 * merge, and the post-processing internally, then returns the merged results.
 *
 * <p>As with {@link CagraIndex#search(CagraQuery)}, the query vectors may be either host- or
 * device-resident; host-resident query matrices are uploaded to the device internally.
 *
 * @since 25.10
 */
public class MultiPartitionCagraSearch {

  private MultiPartitionCagraSearch() {}

  /**
   * Searches multiple CAGRA index partitions for the global top-k nearest neighbors.
   *
   * @param resources shared {@link CuVSResources} handle
   * @param indices   one {@link CagraIndex} per partition, in partition order
   * @param query     a single {@link CagraQuery} whose query matrix is searched against every
   *                  partition; its search parameters are shared across all partitions
   * @param k         number of global nearest neighbors to return per query
   */
  public static MultiPartitionSearchResults search(
      CuVSResources resources, List<CagraIndex> indices, CagraQuery query, int k) throws Throwable {
    return search(resources, indices, query, k, /* filters= */ null);
  }

  /**
   * Searches multiple CAGRA index partitions with optional per-partition device-side filters.
   *
   * @param resources shared {@link CuVSResources} handle
   * @param indices   one {@link CagraIndex} per partition, in partition order
   * @param query     a single {@link CagraQuery} whose query matrix is searched against every
   *                  partition
   * @param k         number of global nearest neighbors to return per query
   * @param filters   one filter per partition, in the same order as {@code indices}, or
   *                  {@code null}/empty for a fully unfiltered search. When non-null, its size must
   *                  equal {@code indices.size()}; a {@code null} entry means no filter for that
   *                  partition. Each handle must be obtained from
   *                  {@link FilterBitsetHandle#create(long[])} for that partition's packed bitset;
   *                  handles from other sources are not supported.
   */
  public static MultiPartitionSearchResults search(
      CuVSResources resources,
      List<CagraIndex> indices,
      CagraQuery query,
      int k,
      List<FilterBitsetHandle> filters)
      throws Throwable {
    return CuVSProvider.provider().searchCagraMultiPartition(resources, indices, query, k, filters);
  }
}

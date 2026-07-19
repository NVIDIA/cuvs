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
    return search(resources, indices, query, k, /* filter= */ null);
  }

  /**
   * Searches multiple CAGRA index partitions with an optional pre-cached device-side filter.
   *
   * @param resources shared {@link CuVSResources} handle
   * @param indices   one {@link CagraIndex} per partition, in partition order
   * @param query     a single {@link CagraQuery} whose query matrix is searched against every
   *                  partition
   * @param k         number of global nearest neighbors to return per query
   * @param filter    pre-built combined bitset handle obtained from
   *                  {@link FilterBitsetHandle#create(long[])}, or {@code null} for unfiltered
   *                  search. Handles from other sources are not supported.
   */
  public static MultiPartitionSearchResults search(
      CuVSResources resources,
      List<CagraIndex> indices,
      CagraQuery query,
      int k,
      FilterBitsetHandle filter)
      throws Throwable {
    return CuVSProvider.provider().searchCagraMultiPartition(resources, indices, query, k, filter);
  }
}

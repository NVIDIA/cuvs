/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import com.nvidia.cuvs.spi.CuVSProvider;
import java.io.InputStream;
import java.util.Objects;

/**
 * {@link HnswIndex} encapsulates a HNSW index, along with methods to interact
 * with it.
 *
 * @since 25.02
 */
public interface HnswIndex extends AutoCloseable {

  /**
   * Invokes the native destroy_hnsw_index to de-allocate the HNSW index
   */
  @Override
  void close() throws Exception;

  /**
   * Invokes the native search_hnsw_index via the Panama API for searching a HNSW
   * index.
   *
   * @param query an instance of {@link HnswQuery} holding the query vectors and
   *              other parameters
   * @return an instance of {@link SearchResults} containing the results
   */
  SearchResults search(HnswQuery query) throws Throwable;

  /**
   * Creates a new Builder with an instance of {@link CuVSResources}.
   *
   * @param cuvsResources an instance of {@link CuVSResources}
   * @throws UnsupportedOperationException if the provider does not cuvs
   */
  static HnswIndex.Builder newBuilder(CuVSResources cuvsResources) {
    Objects.requireNonNull(cuvsResources);
    return CuVSProvider.provider().newHnswIndexBuilder(cuvsResources);
  }

  /**
   * Creates an HNSW index from an existing CAGRA index.
   *
   * @param hnswParams Parameters for the HNSW index
   * @param cagraIndex The CAGRA index to convert from
   * @return A new HNSW index
   * @throws Throwable if an error occurs during conversion
   */
  static HnswIndex fromCagra(HnswIndexParams hnswParams, CagraIndex cagraIndex) throws Throwable {
    Objects.requireNonNull(hnswParams);
    Objects.requireNonNull(cagraIndex);
    return CuVSProvider.provider().hnswIndexFromCagra(hnswParams, cagraIndex);
  }

  /**
   * Builds an HNSW index from HNSW parameters. The graph is built on the GPU and converted to an
   * HNSW index that can be searched on the CPU. The graph build algorithm is selected automatically
   * unless explicit ACE parameters are provided.
   *
   * @param resources The CuVS resources
   * @param hnswParams Parameters for the HNSW index
   * @param dataset The dataset to build the index from
   * @return A new HNSW index ready for search
   * @throws Throwable if an error occurs during building
   */
  static HnswIndex build(CuVSResources resources, HnswIndexParams hnswParams, CuVSMatrix dataset)
      throws Throwable {
    Objects.requireNonNull(resources);
    Objects.requireNonNull(hnswParams);
    Objects.requireNonNull(dataset);
    return CuVSProvider.provider().hnswIndexBuild(resources, hnswParams, dataset);
  }

  /**
   * Builder helps configure and create an instance of {@link HnswIndex}.
   */
  interface Builder {

    /**
     * Sets an instance of InputStream typically used when index deserialization is
     * needed.
     *
     * @param inputStream an instance of {@link InputStream}
     * @return an instance of this Builder
     */
    Builder from(InputStream inputStream);

    /**
     * Registers an instance of configured {@link HnswIndexParams} with this
     * Builder.
     *
     * @param hnswIndexParameters An instance of HnswIndexParams.
     * @return An instance of this Builder.
     */
    Builder withIndexParams(HnswIndexParams hnswIndexParameters);

    /**
     * Builds and returns an instance of CagraIndex.
     *
     * @return an instance of CagraIndex
     */
    HnswIndex build() throws Throwable;
  }
}

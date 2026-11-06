/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import com.nvidia.cuvs.TieredIndex.TieredIndexType;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.Objects;

/**
 * TieredIndexQuery holds the search parameters and query vectors to be used
 * while invoking search. Currently only supports CAGRA index type.
 *
 * <p><strong>Thread Safety:</strong> Each TieredIndexQuery instance should use its own
 * CuVSResources object that is not shared with other threads. Sharing CuVSResources
 * between threads can lead to memory allocation errors or JVM crashes.
 *
 * @since 25.02
 */
public class TieredIndexQuery {
  private TieredIndexType indexType;
  private CagraSearchParams cagraSearchParameters;
  private List<Integer> mapping;
  private float[][] queryVectors;
  private int topK;
  private BitSet prefilter;
  private long numDocs;
  private final CuVSResources resources;

  private TieredIndexQuery(
      TieredIndexType indexType,
      CagraSearchParams cagraSearchParameters,
      List<Integer> mapping,
      float[][] queryVectors,
      int topK,
      BitSet prefilter,
      long numDocs,
      CuVSResources resources) {
    super();
    this.indexType = indexType;
    this.cagraSearchParameters = cagraSearchParameters;
    this.mapping = mapping;
    this.queryVectors = queryVectors;
    this.topK = topK;
    this.prefilter = prefilter;
    this.numDocs = numDocs;
    this.resources = resources;
  }

  /**
   * Gets the index type for this query.
   *
   * @return the TieredIndexType
   */
  public TieredIndexType getIndexType() {
    return indexType;
  }

  /**
   * Gets the instance of CagraSearchParams initially set.
   *
   * @return an instance CagraSearchParams
   */
  public CagraSearchParams getCagraSearchParameters() {
    return cagraSearchParameters;
  }

  /**
   * Gets the query vector 2D float array.
   *
   * @return 2D float array
   */
  public float[][] getQueryVectors() {
    return queryVectors;
  }

  /**
   * Gets the passed map instance.
   *
   * @return a map of ID mappings
   */
  public List<Integer> getMapping() {
    return mapping;
  }

  /**
   * Gets the topK value.
   *
   * @return the topK value
   */
  public int getTopK() {
    return topK;
  }

  /**
   * Gets the prefilter BitSet.
   *
   * @return a BitSet object representing the prefilter
   */
  public BitSet getPrefilter() {
    return prefilter;
  }

  /**
   * Gets the number of documents in this index, as used for prefilter.
   *
   * @return number of documents as an integer
   */
  public long getNumDocs() {
    return numDocs;
  }

  /**
   * Gets the CuVSResources instance for this query.
   *
   * @return the CuVSResources instance
   */
  public CuVSResources getResources() {
    return resources;
  }

  @Override
  public String toString() {
    return "TieredIndexQuery [indexType="
        + indexType
        + ", cagraSearchParameters="
        + cagraSearchParameters
        + ", queryVectors="
        + Arrays.toString(queryVectors)
        + ", mapping="
        + mapping
        + ", topK="
        + topK
        + "]";
  }

  /**
   * Creates a new Builder instance.
   *
   * @param resources the CuVSResources instance to use for this query
   * @return a new Builder instance
   */
  public static Builder newBuilder(CuVSResources resources) {
    return new Builder(resources);
  }

  /**
   * Builder helps configure and create an instance of TieredIndexQuery.
   */
  public static class Builder {
    private TieredIndexType indexType = TieredIndexType.CAGRA;
    private CagraSearchParams cagraSearchParams;
    private float[][] queryVectors;
    private List<Integer> mapping;
    private int topK = 2;
    private BitSet prefilter;
    private long numDocs;
    private final CuVSResources resources;

    /**
     * Constructor that requires CuVSResources.
     *
     * <p><strong>Important:</strong> The provided CuVSResources instance should not be
     * shared with other threads. Each thread performing searches should create its own
     * CuVSResources instance to avoid memory allocation conflicts and potential JVM crashes.
     *
     * @param resources the CuVSResources instance to use for this query (must not be shared between threads)
     */
    public Builder(CuVSResources resources) {
      this.resources = Objects.requireNonNull(resources, "resources cannot be null");
    }

    /**
     * Sets the index type for this query.
     *
     * @param indexType the index type
     * @return an instance of this Builder
     */
    public Builder withIndexType(TieredIndexType indexType) {
      this.indexType = indexType;
      return this;
    }

    /**
     * Sets the instance of configured CagraSearchParams to be passed for search.
     *
     * @param cagraSearchParams an instance of the configured CagraSearchParams to
     *                          be used for this query
     * @return an instance of this Builder
     */
    public Builder withSearchParams(CagraSearchParams cagraSearchParams) {
      this.cagraSearchParams = cagraSearchParams;
      return this;
    }

    /**
     * Registers the query vectors to be passed in the search call.
     *
     * @param queryVectors 2D float query vector array
     * @return an instance of this Builder
     */
    public Builder withQueryVectors(float[][] queryVectors) {
      this.queryVectors = queryVectors;
      return this;
    }

    /**
     * Sets the instance of mapping to be used for ID mapping.
     *
     * @param mapping the ID mapping instance
     * @return an instance of this Builder
     */
    public Builder withMapping(List<Integer> mapping) {
      this.mapping = mapping;
      return this;
    }

    /**
     * Registers the topK value.
     *
     * @param topK the topK value used to retrieve the topK results
     * @return an instance of this Builder
     */
    public Builder withTopK(int topK) {
      this.topK = topK;
      return this;
    }

    /**
     * Sets a prefilter shared by all queries. Bit {@code i} set includes dataset vector {@code i}
     * in the results; bit {@code i} clear excludes it. A {@code BitSet} with no bits set excludes
     * every vector, so the search returns no results, while a {@code null} prefilter applies no
     * filtering.
     *
     * <p>Known limitation: a filtered search of an index that holds both vectors in its CAGRA
     * index and vectors added by {@code extend} that have not been indexed yet can return wrong
     * results: cuVS filters the latter by the wrong bits, and can return wrong ids and distances.
     *
     * @param prefilter the filter, or {@code null} for no filtering
     * @param numDocs the number of vectors in the index at the time of the search, including any
     *     added by {@code extend}. cuVS reads one bit per vector, so a smaller value leaves it
     *     reading past the end of the filter. Bits at or beyond {@code numDocs} are ignored.
     * @return an instance of this Builder
     */
    public Builder withPrefilter(BitSet prefilter, int numDocs) {
      this.prefilter = prefilter;
      this.numDocs = numDocs;
      return this;
    }

    /**
     * Builds an instance of TieredIndexQuery.
     *
     * @return an instance of TieredIndexQuery
     * @throws IllegalStateException if required parameters are missing
     */
    public TieredIndexQuery build() {
      return new TieredIndexQuery(
          indexType, cagraSearchParams, mapping, queryVectors, topK, prefilter, numDocs, resources);
    }
  }
}

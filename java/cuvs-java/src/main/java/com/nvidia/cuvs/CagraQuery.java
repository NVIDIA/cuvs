/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import java.util.BitSet;
import java.util.Objects;
import java.util.function.LongToIntFunction;

/**
 * CagraQuery holds the CagraSearchParams and the query vectors to be used while
 * invoking search.
 *
 * <p><strong>Thread Safety:</strong> Each CagraQuery instance should use its own
 * CuVSResources object that is not shared with other threads. Sharing CuVSResources
 * between threads can lead to memory allocation errors or JVM crashes.
 *
 * @since 25.02
 */
public class CagraQuery {

  private final CagraSearchParams cagraSearchParameters;
  private final LongToIntFunction mapping;
  private final CuVSMatrix queryVectors;
  private final int topK;
  private final BitSet prefilter;
  private final int numDocs;
  private final CuVSResources resources;

  /**
   * Constructs an instance of {@link CagraQuery} using cagraSearchParameters,
   * preFilter, queryVectors, mapping, and topK.
   *
   * @param cagraSearchParameters an instance of {@link CagraSearchParams} holding
   *                              the search parameters
   * @param queryVectors          2D float query vector array
   * @param mapping               a function mapping ordinals (neighbor IDs) to custom user IDs
   * @param topK                  the top k results to return
   * @param prefilter             A single BitSet to use as filter while searching the CAGRA index
   * @param numDocs               Total number of dataset vectors; used to align the prefilter correctly
   * @param resources             CuVSResources instance to use for this query
   */
  private CagraQuery(
      CagraSearchParams cagraSearchParameters,
      CuVSMatrix queryVectors,
      LongToIntFunction mapping,
      int topK,
      BitSet prefilter,
      int numDocs,
      CuVSResources resources) {
    super();
    this.cagraSearchParameters = cagraSearchParameters;
    this.queryVectors = queryVectors;
    this.mapping = mapping;
    this.topK = topK;
    this.prefilter = prefilter;
    this.numDocs = numDocs;
    this.resources = resources;
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
   * Gets the query vector matrix.
   */
  public CuVSMatrix getQueryVectors() {
    return queryVectors;
  }

  /**
   * Gets the function mapping ordinals (neighbor IDs) to custom user IDs
   */
  public LongToIntFunction getMapping() {
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
   * Gets the number of documents in this index, as used for prefilter
   *
   * @return number of documents as an integer
   */
  public int getNumDocs() {
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
    return "CuVSQuery [cagraSearchParameters="
        + cagraSearchParameters
        + ", queryVectors="
        + queryVectors.toString()
        + ", mapping="
        + mapping
        + ", topK="
        + topK
        + "]";
  }

  /**
   * Builder helps configure and create an instance of CagraQuery.
   */
  public static class Builder {

    private CagraSearchParams cagraSearchParams;
    private CuVSMatrix queryVectors;
    private LongToIntFunction mapping = SearchResults.IDENTITY_MAPPING;
    private int topK = 2;
    private BitSet prefilter;
    private int numDocs;
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
     * @param queryVectors 2D query vector array
     * @return an instance of this Builder
     */
    public Builder withQueryVectors(CuVSMatrix queryVectors) {
      this.queryVectors = queryVectors;
      return this;
    }

    /**
     * Sets the function used to map ordinals (neighbor IDs) to custom user IDs
     *
     * @param mapping a function mapping ordinals (neighbor IDs) to custom user IDs
     * @return an instance of this Builder
     */
    public Builder withMapping(LongToIntFunction mapping) {
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
     * Sets a prefilter shared by all queries in this {@link CagraQuery}. Bit {@code i} set
     * includes dataset vector {@code i} in the results; bit {@code i} clear excludes it. A
     * {@code BitSet} with no bits set excludes every vector, so the search returns no results,
     * while a {@code null} prefilter applies no filtering.
     *
     * @param prefilter the filter, or {@code null} for no filtering
     * @param numDocs the number of vectors the filter covers. The search throws {@link
     *     IllegalArgumentException} if it is smaller than the number of vectors in the index. Bits
     *     at or beyond {@code numDocs} are ignored.
     * @return this {@link Builder} instance
     */
    public Builder withPrefilter(BitSet prefilter, int numDocs) {
      this.prefilter = prefilter;
      this.numDocs = numDocs;
      return this;
    }

    /**
     * Builds an instance of CuVSQuery.
     *
     * @return an instance of CuVSQuery
     */
    public CagraQuery build() {
      return new CagraQuery(
          cagraSearchParams, queryVectors, mapping, topK, prefilter, numDocs, resources);
    }
  }
}

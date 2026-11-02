/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import java.util.Arrays;
import java.util.BitSet;
import java.util.Objects;
import java.util.function.LongToIntFunction;

/**
 * BruteForceQuery holds the query vectors to be used while invoking search.
 *
 * <p><strong>Thread Safety:</strong> Each BruteForceQuery instance should use its own
 * CuVSResources object that is not shared with other threads. Sharing CuVSResources
 * between threads can lead to memory allocation errors or JVM crashes.
 *
 * @since 25.02
 */
public class BruteForceQuery {

  private final LongToIntFunction mapping;
  private final float[][] queryVectors;
  private final BitSet[] prefilters;
  private int numDocs = -1;
  private final int topK;
  private final CuVSResources resources;

  /**
   * Constructs an instance of {@link BruteForceQuery} using queryVectors,
   * mapping, and topK.
   *
   * @param queryVectors 2D float query vector array
   * @param mapping      a function mapping ordinals (neighbor IDs) to custom user IDs
   * @param topK         the top k results to return
   * @param prefilters   the prefilters data to use while searching the BRUTEFORCE
   *                     index
   * @param numDocs      Maximum of bits in each prefilter, representing number of documents in this index.
   *                     Used only when prefilter(s) is/are passed.
   * @param resources    CuVSResources instance to use for this query
   */
  public BruteForceQuery(
      float[][] queryVectors,
      LongToIntFunction mapping,
      int topK,
      BitSet[] prefilters,
      int numDocs,
      CuVSResources resources) {
    this.queryVectors = queryVectors;
    this.mapping = mapping;
    this.topK = topK;
    this.prefilters = prefilters;
    this.numDocs = numDocs;
    this.resources = resources;
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
   * Gets the function mapping ordinals (neighbor IDs) to custom user IDs
   */
  public LongToIntFunction getMapping() {
    return mapping;
  }

  /**
   * Gets the topK value.
   *
   * @return an integer
   */
  public int getTopK() {
    return topK;
  }

  /**
   * Gets the prefilter long array
   *
   * @return an array of bitsets
   */
  public BitSet[] getPrefilters() {
    return prefilters;
  }

  /**
   * Gets the number of documents supposed to be in this index, as used for prefilters
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
    return "BruteForceQuery [mapping="
        + mapping
        + ", queryVectors="
        + Arrays.toString(queryVectors)
        + ", prefilter="
        + Arrays.toString(prefilters)
        + ", topK="
        + topK
        + "]";
  }

  /**
   * Builder helps configure and create an instance of BruteForceQuery.
   */
  public static class Builder {

    private float[][] queryVectors;
    private BitSet[] prefilters;
    private int numDocs;
    private LongToIntFunction mapping = SearchResults.IDENTITY_MAPPING;
    private int topK = 2;
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
     * Sets the prefilters for building the {@link BruteForceQuery}. Bit {@code i} set includes
     * dataset vector {@code i} in the results; bit {@code i} clear excludes it.
     *
     * <p>Pass either one {@code BitSet}, which applies to every query, or one per query; bitsets
     * beyond the number of queries are ignored. A {@code BitSet} with no bits set excludes every
     * vector, so its queries return no results, while a {@code null} entry applies no filtering
     * to its query. A {@code null} or empty array applies no filtering at all. The search throws
     * {@link IllegalArgumentException} if there are several bitsets but fewer than queries.
     *
     * @param prefilters one bitset shared by every query, or one per query
     * @param numDocs the number of vectors in the index, which each bitset covers. It must equal
     *     that number: cuVS reads each bitset over the index's rows, and the bitsets of a
     *     per-query array are placed {@code numDocs} bits apart, so any other value reads the
     *     wrong bits or past the end of the filter. Bits at or beyond {@code numDocs} are ignored.
     * @return an instance of this Builder
     */
    public Builder withPrefilters(BitSet[] prefilters, int numDocs) {
      this.prefilters = prefilters;
      this.numDocs = numDocs;
      return this;
    }

    /**
     * Builds an instance of {@link BruteForceQuery}
     *
     * @return an instance of {@link BruteForceQuery}
     */
    public BruteForceQuery build() {
      return new BruteForceQuery(queryVectors, mapping, topK, prefilters, numDocs, resources);
    }
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

/**
 * CagraSearchParams encapsulates the logic for configuring and holding search
 * parameters.
 *
 * @since 25.02
 */
public class CagraSearchParams {

  private int maxQueries;
  private int iTopKSize;
  private int maxIterations;
  private int teamSize;
  private int searchWidth;
  private int minIterations;
  private int threadBlockSize;
  private int hashmapMinBitlen;
  private int numRandomSamplings;
  private float hashMapMaxFillRate;
  private long randXORMask;
  private SearchAlgo searchAlgo;
  private HashMapMode hashMapMode;

  /**
   * Enum to denote algorithm used to search CAGRA Index.
   */
  public enum SearchAlgo {
    /**
     * for large batch sizes
     */
    SINGLE_CTA(0),
    /**
     * for small batch sizes
     */
    MULTI_CTA(1),
    /**
     * MULTI_KERNEL
     */
    MULTI_KERNEL(2),
    /**
     * AUTO
     */
    AUTO(100);

    /**
     * The value for the enum choice.
     */
    public final int value;

    private SearchAlgo(int value) {
      this.value = value;
    }
  }

  /**
   * Enum to denote Hash Mode used while searching CAGRA index.
   */
  public enum HashMapMode {
    /**
     * HASH
     */
    HASH(0),
    /**
     * SMALL
     */
    SMALL(1),
    /**
     * AUTO_HASH
     */
    AUTO_HASH(100);

    /**
     * The value for the enum choice.
     */
    public final int value;

    private HashMapMode(int value) {
      this.value = value;
    }
  }

  /**
   * Constructs an instance of CagraSearchParams with passed search parameters.
   *
   * @param maxQueries         the maximum number of queries to search at the same
   *                           time (batch size)
   * @param iTopKSize          the number of intermediate search results retained
   *                           during the search
   * @param maxIterations      the upper limit of search iterations
   * @param searchAlgo         the search implementation is configured
   * @param teamSize           the number of threads used to calculate a single
   *                           distance
   * @param searchWidth        the number of graph nodes to select as the starting
   *                           point for the search in each iteration
   * @param minIterations      the lower limit of search iterations
   * @param threadBlockSize    the thread block size
   * @param hashmapMode        the hash map type configured
   * @param hashmapMinBitlen   the lower limit of hash map bit length
   * @param hashmapMaxFillRate the upper limit of hash map fill rate
   * @param numRandomSamplings the number of iterations of initial random seed
   *                           node selection
   * @param randXORMask        the bit mask used for initial random seed node
   *                           selection
   */
  private CagraSearchParams(
      int maxQueries,
      int iTopKSize,
      int maxIterations,
      SearchAlgo searchAlgo,
      int teamSize,
      int searchWidth,
      int minIterations,
      int threadBlockSize,
      HashMapMode hashmapMode,
      int hashmapMinBitlen,
      float hashmapMaxFillRate,
      int numRandomSamplings,
      long randXORMask) {
    this.maxQueries = maxQueries;
    this.iTopKSize = iTopKSize;
    this.maxIterations = maxIterations;
    this.searchAlgo = searchAlgo;
    this.teamSize = teamSize;
    this.searchWidth = searchWidth;
    this.minIterations = minIterations;
    this.threadBlockSize = threadBlockSize;
    this.hashMapMode = hashmapMode;
    this.hashmapMinBitlen = hashmapMinBitlen;
    this.hashMapMaxFillRate = hashmapMaxFillRate;
    this.numRandomSamplings = numRandomSamplings;
    this.randXORMask = randXORMask;
  }

  /**
   * Gets the maximum number of queries to search at the same time (batch size).
   *
   * @return the maximum number of queries
   */
  public int getMaxQueries() {
    return maxQueries;
  }

  /**
   * Gets the number of intermediate search results retained during the search.
   *
   * @return the number of intermediate search results
   */
  public int getITopKSize() {
    return iTopKSize;
  }

  /**
   * Gets the upper limit of search iterations.
   *
   * @return the upper limit value
   */
  public int getMaxIterations() {
    return maxIterations;
  }

  /**
   * Gets the number of threads used to calculate a single distance.
   *
   * @return the number of threads configured
   */
  public int getTeamSize() {
    return teamSize;
  }

  /**
   * Gets the number of graph nodes to select as the starting point for the search
   * in each iteration.
   *
   * @return the number of graph nodes
   */
  public int getSearchWidth() {
    return searchWidth;
  }

  /**
   * Gets the lower limit of search iterations.
   *
   * @return the lower limit value
   */
  public int getMinIterations() {
    return minIterations;
  }

  /**
   * Gets the thread block size.
   *
   * @return the thread block size
   */
  public int getThreadBlockSize() {
    return threadBlockSize;
  }

  /**
   * Gets the lower limit of hash map bit length.
   *
   * @return the lower limit value
   */
  public int getHashmapMinBitlen() {
    return hashmapMinBitlen;
  }

  /**
   * Gets the number of iterations of initial random seed node selection.
   *
   * @return the number of iterations
   */
  public int getNumRandomSamplings() {
    return numRandomSamplings;
  }

  /**
   * Gets the upper limit of hash map fill rate.
   *
   * @return the upper limit of hash map fill rate
   */
  public float getHashMapMaxFillRate() {
    return hashMapMaxFillRate;
  }

  /**
   * Gets the bit mask used for initial random seed node selection.
   *
   * @return the bit mask value
   */
  public long getRandXORMask() {
    return randXORMask;
  }

  /**
   * Gets which search implementation is configured.
   *
   * @return the configured {@link SearchAlgo}
   */
  public SearchAlgo getCagraSearchAlgo() {
    return searchAlgo;
  }

  /**
   * Gets the hash map mode configured.
   *
   * @return the configured {@link HashMapMode}
   */
  public HashMapMode getHashMapMode() {
    return hashMapMode;
  }

  @Override
  public String toString() {
    return "CagraSearchParams [maxQueries="
        + maxQueries
        + ", itopkSize="
        + iTopKSize
        + ", maxIterations="
        + maxIterations
        + ", cuvsCagraSearchAlgo="
        + searchAlgo
        + ", teamSize="
        + teamSize
        + ", searchWidth="
        + searchWidth
        + ", minIterations="
        + minIterations
        + ", threadBlockSize="
        + threadBlockSize
        + ", hashMapMode="
        + hashMapMode
        + ", hashMapMinBitlen="
        + hashmapMinBitlen
        + ", hashMapMaxFillRate="
        + hashMapMaxFillRate
        + ", numRandomSamplings="
        + numRandomSamplings
        + ", randXORMask="
        + randXORMask
        + "]";
  }

  /**
   * Builder configures and creates an instance of CagraSearchParams.
   */
  public static class Builder {

    private int maxQueries;
    private int iTopKSize = 64;
    private int maxIterations;
    private int teamSize;
    private int searchWidth = 1;
    private int minIterations;
    private int threadBlockSize;
    private int hashMapMinBitlen;
    private int numRandomSamplings = 1;
    private float hashMapMaxFillRate = 0.5f;
    private long randXORMask = 0x128394;
    private SearchAlgo searchAlgo = SearchAlgo.AUTO;
    private HashMapMode hashMapMode = HashMapMode.AUTO_HASH;

    /**
     * Default constructor.
     */
    public Builder() {}

    /**
     * Sets the maximum number of queries to search at the same time (batch size).
     * Auto select when 0.
     *
     * @param maxQueries the maximum number of queries
     * @return an instance of this Builder
     */
    public Builder withMaxQueries(int maxQueries) {
      this.maxQueries = maxQueries;
      return this;
    }

    /**
     * Sets the number of intermediate search results retained during the search.
     * This is the main knob to adjust trade off between accuracy and search speed.
     * Higher values improve the search accuracy.
     *
     * @param iTopKSize the number of intermediate search results
     * @return an instance of this Builder
     */
    public Builder withItopkSize(int iTopKSize) {
      this.iTopKSize = iTopKSize;
      return this;
    }

    /**
     * Sets the upper limit of search iterations. Auto select when 0.
     *
     * @param maxIterations the upper limit of search iterations
     * @return an instance of this Builder
     */
    public Builder withMaxIterations(int maxIterations) {
      this.maxIterations = maxIterations;
      return this;
    }

    /**
     * Sets which search implementation to use.
     *
     * @param cuvsCagraSearchAlgo the {@link SearchAlgo} to use
     * @return an instance of this Builder
     */
    public Builder withAlgo(SearchAlgo cuvsCagraSearchAlgo) {
      this.searchAlgo = cuvsCagraSearchAlgo;
      return this;
    }

    /**
     * Sets the number of threads used to calculate a single distance. 4, 8, 16, or
     * 32.
     *
     * @param teamSize the number of threads used to calculate a single distance
     * @return an instance of this Builder
     */
    public Builder withTeamSize(int teamSize) {
      this.teamSize = teamSize;
      return this;
    }

    /**
     * Sets the number of graph nodes to select as the starting point for the search
     * in each iteration.
     *
     * @param searchWidth the number of graph nodes to select
     * @return an instance of this Builder
     */
    public Builder withSearchWidth(int searchWidth) {
      this.searchWidth = searchWidth;
      return this;
    }

    /**
     * Sets the lower limit of search iterations.
     *
     * @param minIterations the lower limit of search iterations
     * @return an instance of this Builder
     */
    public Builder withMinIterations(int minIterations) {
      this.minIterations = minIterations;
      return this;
    }

    /**
     * Sets the thread block size. 0, 64, 128, 256, 512, 1024. Auto selection when
     * 0.
     *
     * @param threadBlockSize the thread block size
     * @return an instance of this Builder
     */
    public Builder withThreadBlockSize(int threadBlockSize) {
      this.threadBlockSize = threadBlockSize;
      return this;
    }

    /**
     * Sets the hash map type. Auto selection when AUTO.
     *
     * @param hashMapMode the {@link HashMapMode}
     * @return an instance of this Builder
     */
    public Builder withHashMapMode(HashMapMode hashMapMode) {
      this.hashMapMode = hashMapMode;
      return this;
    }

    /**
     * Sets the lower limit of hash map bit length. More than 8.
     *
     * @param hashMapMinBitlen the lower limit of hash map bit length
     * @return an instance of this Builder
     */
    public Builder withHashMapMinBitlen(int hashMapMinBitlen) {
      this.hashMapMinBitlen = hashMapMinBitlen;
      return this;
    }

    /**
     * Sets the upper limit of hash map fill rate. More than 0.1, less than 0.9.
     *
     * @param hashMapMaxFillRate the upper limit of hash map fill rate
     * @return an instance of this Builder
     */
    public Builder withHashMapMaxFillRate(float hashMapMaxFillRate) {
      this.hashMapMaxFillRate = hashMapMaxFillRate;
      return this;
    }

    /**
     * Sets the number of iterations of initial random seed node selection. 1 or
     * more.
     *
     * @param numRandomSamplings the number of iterations of initial random seed
     *                           node selection
     * @return an instance of this Builder
     */
    public Builder withNumRandomSamplings(int numRandomSamplings) {
      this.numRandomSamplings = numRandomSamplings;
      return this;
    }

    /**
     * Sets the bit mask used for initial random seed node selection.
     *
     * @param randXORMask the bit mask used for initial random seed node selection
     * @return an instance of this Builder
     */
    public Builder withRandXorMask(long randXORMask) {
      this.randXORMask = randXORMask;
      return this;
    }

    /**
     * Builds an instance of {@link CagraSearchParams} with passed search
     * parameters.
     *
     * @return an instance of CagraSearchParams
     */
    public CagraSearchParams build() {
      return new CagraSearchParams(
          maxQueries,
          iTopKSize,
          maxIterations,
          searchAlgo,
          teamSize,
          searchWidth,
          minIterations,
          threadBlockSize,
          hashMapMode,
          hashMapMinBitlen,
          hashMapMaxFillRate,
          numRandomSamplings,
          randXORMask);
    }
  }
}

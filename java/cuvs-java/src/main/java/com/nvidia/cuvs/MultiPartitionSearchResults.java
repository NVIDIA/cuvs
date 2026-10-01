/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

/**
 * Holds the decoded results of a multi-partition GPU search.
 *
 * <p>Each entry {@code i} in [0, {@link #count}) identifies:
 * <ul>
 *   <li>which input partition the result came from ({@link #getPartitionIndex(int)})</li>
 *   <li>the local vector ordinal within that partition ({@link #getOrdinal(int)})</li>
 *   <li>the raw CAGRA distance ({@link #getDistance(int)})</li>
 * </ul>
 *
 * <p>The caller is responsible for mapping ordinals to its own global identifiers.
 *
 * @since 25.10
 */
public class MultiPartitionSearchResults {

  private final int count;
  private final int[] partitionIndices;
  private final int[] ordinals;
  private final float[] distances;

  public MultiPartitionSearchResults(
      int count, int[] partitionIndices, int[] ordinals, float[] distances) {
    this.count = count;
    this.partitionIndices = partitionIndices;
    this.ordinals = ordinals;
    this.distances = distances;
  }

  /** Number of valid results (may be less than k if fewer candidates exist). */
  public int count() {
    return count;
  }

  /** Index into the original partition list for result {@code i}. */
  public int getPartitionIndex(int i) {
    return partitionIndices[i];
  }

  /** Local vector ordinal within the partition for result {@code i}. */
  public int getOrdinal(int i) {
    return ordinals[i];
  }

  /** Post-processed distance for result {@code i} (scaled + metric-transformed). */
  public float getDistance(int i) {
    return distances[i];
  }
}

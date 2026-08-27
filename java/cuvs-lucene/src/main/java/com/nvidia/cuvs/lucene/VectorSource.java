/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.Closeable;
import java.io.IOException;

/**
 * A forward-only, single-consumer source of float vectors for {@link CagraHnswBulkIndexWriter}.
 *
 * <p>Implementations own how the underlying data is fetched (a local file, a database cursor, an
 * object-store stream, ...); {@link CagraHnswBulkIndexWriter} only depends on this contract, not on
 * any particular storage.
 *
 * <p>{@link #get(int, float[])} must be called with non-decreasing indices from a single thread.
 * This mirrors how a bulk indexer consumes its input (front-to-back, one pass) and lets
 * implementations use a simple prefetch/streaming strategy instead of arbitrary random access.
 */
public interface VectorSource extends Closeable {

  /** Number of dimensions of every vector returned by this source. */
  int dimensions();

  /** Number of vectors available, i.e. the exclusive upper bound on indices passed to {@link #get}. */
  int size();

  /**
   * Fills {@code dst} with the vector at {@code index} (no allocation). {@code index} must be
   * greater than or equal to the index passed to the previous call, if any.
   */
  void get(int index, float[] dst) throws IOException;
}

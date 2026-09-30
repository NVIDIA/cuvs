/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.LibraryException;

/**
 * cuVS based Scalar Quantized KnnVectorsFormat for indexing on GPU and searching on the CPU.
 *
 * <p>Stores the vectors with Lucene's {@code Lucene99ScalarQuantizedVectorsFormat}, which Lucene
 * 10.4 moved to its backward codecs. On Lucene 10.4 and later this format can only read existing
 * indexes; use {@link CuVSCodecs#acceleratedHNSWScalarQuantizedFormat} to write.
 *
 * @since 26.02
 */
public class LuceneAcceleratedHNSWScalarQuantizedVectorsFormat
    extends BaseAcceleratedHNSWScalarQuantizedVectorsFormat {

  /**
   * Initializes {@link LuceneAcceleratedHNSWScalarQuantizedVectorsFormat} with default values.
   *
   * @throws LibraryException if the native library fails to load
   */
  public LuceneAcceleratedHNSWScalarQuantizedVectorsFormat() {
    this(new AcceleratedHNSWParams.Builder().build());
  }

  /**
   * Initializes {@link LuceneAcceleratedHNSWScalarQuantizedVectorsFormat} with the given threads, graph degree, etc.
   *
   * @param acceleratedHNSWParams An instance of {@link AcceleratedHNSWParams}
   */
  public LuceneAcceleratedHNSWScalarQuantizedVectorsFormat(
      AcceleratedHNSWParams acceleratedHNSWParams) {
    super(
        "Lucene99AcceleratedHNSWScalarQuantizedVectorsFormat",
        acceleratedHNSWParams,
        () -> LuceneCompat.lucene99ScalarQuantizedFlatFormat(),
        params -> LuceneCompat.lucene99HnswScalarQuantizedFormat(params));
  }
}

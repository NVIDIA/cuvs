/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

/**
 * cuVS based scalar-quantized KnnVectorsFormat for indexing on the GPU and searching on the CPU.
 *
 * <p>Stores the vectors with Lucene's {@code Lucene104ScalarQuantizedVectorsFormat}, quantized to 7
 * bits per dimension, which replaced {@code Lucene99ScalarQuantizedVectorsFormat} in Lucene 10.4.
 * The graph is built on the GPU over the vectors quantized the same way as {@link
 * LuceneAcceleratedHNSWScalarQuantizedVectorsFormat} does.
 *
 * @since 26.12
 */
public class Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat
    extends BaseAcceleratedHNSWScalarQuantizedVectorsFormat {

  /** The format's name, which Lucene records in the segments it writes. */
  static final String NAME = "Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat";

  /** Creates the format with default parameters. */
  public Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat() {
    this(new AcceleratedHNSWParams.Builder().build());
  }

  /**
   * Creates the format.
   *
   * @param acceleratedHNSWParams the index build parameters
   */
  public Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat(
      AcceleratedHNSWParams acceleratedHNSWParams) {
    super(
        NAME,
        acceleratedHNSWParams,
        () -> LuceneCompat.lucene104ScalarQuantizedFlatFormat(),
        params -> LuceneCompat.lucene104HnswScalarQuantizedFormat(params));
  }
}

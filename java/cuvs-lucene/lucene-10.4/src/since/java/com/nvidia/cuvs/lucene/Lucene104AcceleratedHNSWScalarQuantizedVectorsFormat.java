/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.util.logging.Logger;

/**
 * cuVS based scalar-quantized KnnVectorsFormat for indexing on the GPU and searching on the CPU.
 *
 * <p>Stores the vectors with Lucene's {@code Lucene104ScalarQuantizedVectorsFormat}, quantized to
 * one byte per dimension ({@code UNSIGNED_BYTE}), which replaced {@code
 * Lucene99ScalarQuantizedVectorsFormat} in Lucene 10.4. On the GPU, the graph is built over the
 * vectors quantized to 7 bits, as {@link LuceneAcceleratedHNSWScalarQuantizedVectorsFormat} does,
 * so the stored encoding mostly affects how the graph is searched on the CPU. One exception: the
 * flat format normalizes the vectors of COSINE fields when it flushes them, so the graph of a
 * flushed segment is built from normalized vectors, and that of a merged segment from the vectors
 * as indexed. Without a GPU, Lucene builds the graph on the CPU, scoring with the stored encoding.
 *
 * @since 26.12
 */
public class Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat
    extends BaseAcceleratedHNSWScalarQuantizedVectorsFormat {

  private static final Logger log =
      Logger.getLogger(Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat.class.getName());

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
        "Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat",
        "Lucene104ScalarQuantizedVectorsFormat",
        log,
        acceleratedHNSWParams,
        () -> LuceneCompat.lucene104ScalarQuantizedFlatFormat());
  }
}

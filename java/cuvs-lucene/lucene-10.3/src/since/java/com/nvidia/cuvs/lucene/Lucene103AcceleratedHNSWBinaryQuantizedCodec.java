/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

/**
 * A codec that builds HNSW graphs on the GPU over binary-quantized vectors, and writes them in
 * Lucene's HNSW format so that they are searched on the CPU. See {@link
 * LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat}.
 *
 * <p>This codec wraps {@code Lucene103Codec}, the default codec of Lucene 10.3. Once a later Lucene
 * release replaces that default, this codec can only read existing indexes; use {@link
 * CuVSCodecs#acceleratedHNSWBinaryQuantized} to write.
 *
 * @since 26.12
 */
public class Lucene103AcceleratedHNSWBinaryQuantizedCodec extends CuVSFilterCodec {

  private static final String NAME = "Lucene103AcceleratedHNSWBinaryQuantizedCodec";

  /** Creates the codec with default parameters. */
  public Lucene103AcceleratedHNSWBinaryQuantizedCodec() {
    this(new AcceleratedHNSWParams.Builder().build());
  }

  /**
   * Creates the codec.
   *
   * @param acceleratedHNSWParams the index build parameters
   */
  public Lucene103AcceleratedHNSWBinaryQuantizedCodec(AcceleratedHNSWParams acceleratedHNSWParams) {
    super(
        NAME,
        103,
        () -> LuceneCompat.lucene103Codec(),
        () -> new LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat(acceleratedHNSWParams));
  }
}

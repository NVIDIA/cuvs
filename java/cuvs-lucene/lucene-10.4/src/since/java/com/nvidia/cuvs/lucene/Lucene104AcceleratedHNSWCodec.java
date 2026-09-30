/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

/**
 * A codec that builds HNSW graphs on the GPU, and writes them in Lucene's HNSW format so that they
 * are searched on the CPU. Falls back to building the graph on the CPU when no GPU or cuVS is
 * available.
 *
 * <p>This codec wraps {@code Lucene104Codec}, the default codec of Lucene 10.4 and 10.5. Once a
 * later Lucene release replaces that default, this codec can only read existing indexes; use {@link
 * CuVSCodecs#acceleratedHNSW} to write.
 *
 * @since 26.12
 */
public class Lucene104AcceleratedHNSWCodec extends CuVSFilterCodec {

  private static final String NAME = "Lucene104AcceleratedHNSWCodec";

  /** Creates the codec with default parameters. */
  public Lucene104AcceleratedHNSWCodec() {
    this(new AcceleratedHNSWParams.Builder().build());
  }

  /**
   * Creates the codec.
   *
   * @param acceleratedHNSWParams the index build parameters
   */
  public Lucene104AcceleratedHNSWCodec(AcceleratedHNSWParams acceleratedHNSWParams) {
    super(
        NAME,
        104,
        () -> LuceneCompat.lucene104Codec(),
        () -> new Lucene99AcceleratedHNSWVectorsFormat(acceleratedHNSWParams));
  }
}

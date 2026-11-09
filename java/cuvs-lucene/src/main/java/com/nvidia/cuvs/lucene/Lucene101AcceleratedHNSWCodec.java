/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import org.apache.lucene.codecs.Codec;

/**
 * A codec that enables GPU-based accelerated HNSW capability and can be used
 * to accelerated indexing using GPUs and search using CPUs. Fallbacks to CPU
 * based indexing when used on a machine without a GPU and/or cuVS.
 *
 * <p>This codec wraps {@code Lucene101Codec}, the default codec of Lucene 10.2. On later Lucene
 * releases it can only read existing indexes; use {@link CuVSCodecs#acceleratedHNSW} to write.
 *
 * @since 25.10
 */
public class Lucene101AcceleratedHNSWCodec extends CuVSFilterCodec {

  private static final String NAME = "Lucene101AcceleratedHNSWCodec";

  /**
   * Default constructor for {@link Lucene101AcceleratedHNSWCodec}.
   */
  public Lucene101AcceleratedHNSWCodec() {
    this(new AcceleratedHNSWParams.Builder().build());
  }

  /**
   * Constructor for {@link Lucene101AcceleratedHNSWCodec}.
   *
   * @param name the codec's name
   * @param delegate the delegate codec to filter
   */
  public Lucene101AcceleratedHNSWCodec(String name, Codec delegate) {
    super(
        name,
        delegate,
        () ->
            new Lucene99AcceleratedHNSWVectorsFormat(new AcceleratedHNSWParams.Builder().build()));
  }

  /**
   * Constructor for {@link Lucene101AcceleratedHNSWCodec}.
   *
   * @param acceleratedHNSWParams instance of {@link AcceleratedHNSWParams}
   */
  public Lucene101AcceleratedHNSWCodec(AcceleratedHNSWParams acceleratedHNSWParams) {
    super(
        NAME,
        101,
        () -> LuceneCompat.lucene101Codec(),
        () -> new Lucene99AcceleratedHNSWVectorsFormat(acceleratedHNSWParams));
  }
}

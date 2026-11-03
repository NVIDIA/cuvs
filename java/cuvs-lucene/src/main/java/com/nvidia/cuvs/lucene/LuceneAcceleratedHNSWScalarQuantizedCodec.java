/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import org.apache.lucene.codecs.Codec;

/**
 * CuVS based codec for GPU based vector search
 *
 * <p>This codec, named {@code Lucene101AcceleratedHNSWScalarQuantizedCodec}, wraps {@code
 * Lucene101Codec}, the default codec of Lucene 10.2. On later Lucene releases it can only read
 * existing indexes; use {@link CuVSCodecs#acceleratedHNSWScalarQuantized} to write.
 *
 * @since 26.02
 */
public class LuceneAcceleratedHNSWScalarQuantizedCodec extends CuVSFilterCodec {

  private static final String NAME = "Lucene101AcceleratedHNSWScalarQuantizedCodec";

  public LuceneAcceleratedHNSWScalarQuantizedCodec() {
    this(new AcceleratedHNSWParams.Builder().build());
  }

  public LuceneAcceleratedHNSWScalarQuantizedCodec(String name, Codec delegate) {
    super(
        name,
        delegate,
        () ->
            new LuceneAcceleratedHNSWScalarQuantizedVectorsFormat(
                new AcceleratedHNSWParams.Builder().build()));
  }

  public LuceneAcceleratedHNSWScalarQuantizedCodec(AcceleratedHNSWParams acceleratedHNSWParams) {
    super(
        NAME,
        101,
        () -> LuceneCompat.lucene101Codec(),
        () -> new LuceneAcceleratedHNSWScalarQuantizedVectorsFormat(acceleratedHNSWParams));
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import org.apache.lucene.codecs.Codec;

/**
 * A codec that builds HNSW graphs on the GPU, and writes them in Lucene's HNSW format so that they
 * are searched on the CPU. Falls back to building the graph on the CPU when no GPU or cuVS is
 * available.
 *
 * <p>This codec wraps {@code Lucene104Codec}, the default codec of Lucene 10.4. Once a later Lucene
 * release replaces that default, this codec can only read existing indexes; use {@link
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

  /**
   * Creates the codec with a delegate chosen by the caller, for applications that wrap a codec of
   * their own. Unlike the other constructors, this never makes the codec read-only: whether it can
   * write on a later Lucene release depends on the delegate, and on its vectors format, which a
   * later Lucene release may stop writing too.
   *
   * @param name the codec's name, which Lucene records in every segment the codec writes
   * @param delegate the codec to filter
   * @param acceleratedHNSWParams the index build parameters
   */
  public Lucene104AcceleratedHNSWCodec(
      String name, Codec delegate, AcceleratedHNSWParams acceleratedHNSWParams) {
    super(name, delegate, () -> new Lucene99AcceleratedHNSWVectorsFormat(acceleratedHNSWParams));
  }
}

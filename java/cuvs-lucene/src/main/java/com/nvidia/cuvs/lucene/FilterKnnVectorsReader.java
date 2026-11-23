/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;

/**
 * Base class of the readers that forward to another {@link KnnVectorsReader}, such as the view of a
 * source segment of a merge in {@code FallbackHnswVectorsWriter}. It forwards the methods that are
 * the same on every supported Lucene release; {@code CompatFilterKnnVectorsReader}, which extends
 * it once per API variant, forwards the others.
 *
 * <p>It does not close the reader it forwards to, which it does not own, and leaves {@code
 * getMergeInstance} and {@code finishMerge} to Lucene's defaults: its subclasses only stand for a
 * source segment inside a merge, whose reader is already the merge instance.
 */
abstract class FilterKnnVectorsReader extends KnnVectorsReader {

  /** The reader this one forwards to. */
  final KnnVectorsReader in;

  /**
   * @param in the reader to forward to
   */
  FilterKnnVectorsReader(KnnVectorsReader in) {
    this.in = in;
  }

  @Override
  public void checkIntegrity() throws IOException {
    in.checkIntegrity();
  }

  @Override
  public FloatVectorValues getFloatVectorValues(String field) throws IOException {
    return in.getFloatVectorValues(field);
  }

  @Override
  public ByteVectorValues getByteVectorValues(String field) throws IOException {
    return in.getByteVectorValues(field);
  }

  @Override
  public void close() {}
}

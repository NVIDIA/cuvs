/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.Closeable;
import java.io.IOException;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;

/**
 * The part of {@link FilterKnnVectorsWriter} whose shape differs between Lucene releases: up to
 * Lucene 10.4, {@code mergeOneField} merges the vectors and builds the graph of the field and
 * returns nothing; since Lucene 10.5, it merges the vectors and returns the work that builds the
 * graph, which Lucene runs once every field has been merged.
 *
 * <p>This variant calls {@link #prepareGraphMerge} around the whole {@code mergeOneField} of the
 * writer it forwards to.
 */
abstract class CompatFilterKnnVectorsWriter extends FilterKnnVectorsWriter {

  /**
   * @param in the writer to forward to, which this one closes
   */
  CompatFilterKnnVectorsWriter(KnnVectorsWriter in) {
    super(in);
  }

  @Override
  public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    try (Closeable undo = prepareGraphMerge(fieldInfo, mergeState)) {
      in.mergeOneField(fieldInfo, mergeState);
    }
  }
}

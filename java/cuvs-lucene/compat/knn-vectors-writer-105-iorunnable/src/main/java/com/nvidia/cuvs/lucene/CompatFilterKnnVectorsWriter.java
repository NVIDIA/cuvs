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
import org.apache.lucene.util.IORunnable;

/**
 * The part of {@link FilterKnnVectorsWriter} whose shape differs between Lucene releases: since
 * Lucene 10.5, {@code mergeOneField} merges the vectors of the field and returns the work that
 * builds its graph, which Lucene runs once every field has been merged; up to Lucene 10.4, it did
 * both and returned nothing.
 *
 * <p>This variant calls {@link #prepareGraphMerge} around the work that the writer it forwards to
 * returns, so that the change does not outlast the field's graph, nor apply while Lucene merges the
 * vectors of the other fields.
 */
abstract class CompatFilterKnnVectorsWriter extends FilterKnnVectorsWriter {

  /**
   * @param in the writer to forward to, which this one closes
   */
  CompatFilterKnnVectorsWriter(KnnVectorsWriter in) {
    super(in);
  }

  @Override
  public IORunnable mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    IORunnable buildGraph = in.mergeOneField(fieldInfo, mergeState);
    if (buildGraph == null) {
      return null;
    }
    return () -> {
      try (Closeable undo = prepareGraphMerge(fieldInfo, mergeState)) {
        buildGraph.run();
      }
    };
  }
}

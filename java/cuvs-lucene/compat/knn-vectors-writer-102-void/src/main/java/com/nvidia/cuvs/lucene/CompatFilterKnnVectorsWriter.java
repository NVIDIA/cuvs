/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.Closeable;
import java.io.IOException;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.Sorter;

/**
 * Base class of the writers that forward to another {@link KnnVectorsWriter}, such as {@code
 * FallbackHnswVectorsWriter}, and change the {@link MergeState} it builds a merged graph from. The
 * shape of a merge differs between releases: up to Lucene 10.4, {@code mergeOneField} merges the
 * vectors and builds the graph of the field and returns nothing; since Lucene 10.5, it merges the
 * vectors and returns the work that builds the graph, which Lucene runs once every field has been
 * merged.
 *
 * <p>This variant calls {@link #prepareGraphMerge} around the whole {@code mergeOneField} of the
 * writer it forwards to.
 */
abstract class CompatFilterKnnVectorsWriter extends KnnVectorsWriter {

  /** The writer this one forwards to. */
  final KnnVectorsWriter in;

  /**
   * @param in the writer to forward to, which this one closes
   */
  CompatFilterKnnVectorsWriter(KnnVectorsWriter in) {
    this.in = in;
  }

  /**
   * Changes {@code mergeState} for the writer this one forwards to while it builds the merged graph
   * of {@code fieldInfo}.
   *
   * @return what undoes the change, closed once the graph is built
   */
  abstract Closeable prepareGraphMerge(FieldInfo fieldInfo, MergeState mergeState)
      throws IOException;

  @Override
  public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
    return in.addField(fieldInfo);
  }

  @Override
  public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
    in.flush(maxDoc, sortMap);
  }

  @Override
  public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    try (Closeable undo = prepareGraphMerge(fieldInfo, mergeState)) {
      in.mergeOneField(fieldInfo, mergeState);
    }
  }

  @Override
  public void finish() throws IOException {
    in.finish();
  }

  @Override
  public void close() throws IOException {
    in.close();
  }

  @Override
  public long ramBytesUsed() {
    return in.ramBytesUsed();
  }
}

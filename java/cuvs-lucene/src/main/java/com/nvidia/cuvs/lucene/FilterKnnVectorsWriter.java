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
 * FallbackHnswVectorsWriter}, and change the {@link MergeState} it builds a merged graph from. It
 * forwards the methods that are the same on every supported Lucene release; {@code
 * CompatFilterKnnVectorsWriter}, which extends it once per API variant, forwards {@code
 * mergeOneField}, whose shape differs between releases, and calls {@link #prepareGraphMerge} around
 * the building of the graph.
 */
abstract class FilterKnnVectorsWriter extends KnnVectorsWriter {

  /** The writer this one forwards to. */
  final KnnVectorsWriter in;

  /**
   * @param in the writer to forward to, which this one closes
   */
  FilterKnnVectorsWriter(KnnVectorsWriter in) {
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

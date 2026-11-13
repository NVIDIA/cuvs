/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.util.IORunnable;

/**
 * Base class of the cuvs-lucene vectors writers that hides a signature change in {@link
 * KnnVectorsWriter}. Since Lucene 10.5, {@code mergeOneField} returns work for Lucene to run once
 * every field has been merged, where earlier releases return nothing. Each variant of this class
 * overrides {@code mergeOneField} of its Lucene release and forwards it to {@link
 * #doMergeOneField}, whose signature is the same on every release.
 *
 * <p>This variant leaves Lucene nothing to run later: the cuvs-lucene writers build the graph of a
 * field while merging it.
 */
abstract class CompatKnnVectorsWriter extends KnnVectorsWriter {

  /**
   * Merges one field, building its index as it does, and returns {@code null}: nothing is left
   * for Lucene to run once every field has been merged. On Lucene 10.2 to 10.4, it returns nothing
   * ({@code void}).
   */
  @Override
  public IORunnable mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    doMergeOneField(fieldInfo, mergeState);
    return null;
  }

  /** Merges one field. */
  abstract void doMergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException;
}

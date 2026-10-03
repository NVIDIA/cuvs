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
 * A {@link KnnVectorsWriter} whose merge method has the same signature on every Lucene release.
 * Since Lucene 10.5, {@code mergeOneField} may return work for Lucene to run after all fields are
 * merged; the cuvs-lucene writers finish each field right away and return none.
 */
abstract class CompatKnnVectorsWriter extends KnnVectorsWriter {

  @Override
  public IORunnable mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    doMergeOneField(fieldInfo, mergeState);
    return null;
  }

  /** Merges the vectors of one field, and finishes all the work for it before returning. */
  protected abstract void doMergeOneField(FieldInfo fieldInfo, MergeState mergeState)
      throws IOException;
}

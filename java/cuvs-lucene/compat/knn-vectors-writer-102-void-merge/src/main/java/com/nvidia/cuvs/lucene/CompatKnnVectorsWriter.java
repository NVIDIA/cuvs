/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;

/**
 * A {@link KnnVectorsWriter} whose merge method has the same signature on every Lucene release.
 * Through Lucene 10.4, {@code mergeOneField} returns nothing.
 */
abstract class CompatKnnVectorsWriter extends KnnVectorsWriter {

  @Override
  public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    doMergeOneField(fieldInfo, mergeState);
  }

  /** Merges the vectors of one field, and finishes all the work for it before returning. */
  protected abstract void doMergeOneField(FieldInfo fieldInfo, MergeState mergeState)
      throws IOException;
}

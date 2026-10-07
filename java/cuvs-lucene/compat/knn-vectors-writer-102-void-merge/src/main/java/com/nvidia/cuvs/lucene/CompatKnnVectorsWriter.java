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
 * Base class of the cuvs-lucene writers that hides a signature change in {@link KnnVectorsWriter}.
 * Through Lucene 10.4, {@code mergeOneField} returns nothing; Lucene 10.5 made it return work to
 * run after all fields are merged. Each variant of this class overrides the {@code mergeOneField}
 * of its Lucene release and forwards it to {@code doMergeOneField}, whose signature is the same on
 * every release, so the shared writers implement {@code doMergeOneField} once.
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

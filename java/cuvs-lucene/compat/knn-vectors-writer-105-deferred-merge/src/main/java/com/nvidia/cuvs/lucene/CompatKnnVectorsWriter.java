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
 * Base class of the cuvs-lucene writers that hides a signature change in {@link KnnVectorsWriter}.
 * Since Lucene 10.5, {@code mergeOneField} may return work to run after all fields are merged,
 * where earlier releases returned nothing. Each variant of this class overrides the
 * {@code mergeOneField} of its Lucene release and forwards it to {@code doMergeOneField}, whose
 * signature is the same on every release, so the shared writers implement it once.
 *
 * <p>This variant returns no deferred work: the cuvs-lucene writers finish each field right away.
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

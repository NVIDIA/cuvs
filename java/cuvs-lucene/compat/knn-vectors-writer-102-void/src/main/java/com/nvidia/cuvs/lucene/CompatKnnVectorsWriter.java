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
 * Base class of the cuvs-lucene vectors writers that hides a signature change in {@link
 * KnnVectorsWriter}. Since Lucene 10.5, {@code mergeOneField} returns work for Lucene to run once
 * every field has been merged, where earlier releases return nothing. Each variant of this class
 * overrides {@code mergeOneField} of its Lucene release and forwards it to {@link
 * #doMergeOneField}, whose signature is the same on every release.
 */
abstract class CompatKnnVectorsWriter extends KnnVectorsWriter {

  /** Merges one field, building its index as it does. */
  @Override
  public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    doMergeOneField(fieldInfo, mergeState);
  }

  /** Merges one field. */
  abstract void doMergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException;
}

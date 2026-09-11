/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.printInfoStream;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.closeCuVSResourcesInstance;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter.DocMap;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.RamUsageEstimator;

/** Bulk-only writer for vectors already resident in a caller-owned native FBIN mapping. */
final class BorrowedDatasetHnswVectorsWriter extends KnnVectorsWriter {

  private static final long SHALLOW_SIZE =
      RamUsageEstimator.shallowSizeOfInstance(BorrowedDatasetHnswVectorsWriter.class);

  private final BulkIndexingContext context;
  private final InfoStream infoStream;
  private final List<BorrowedDatasetFieldWriter> fields = new ArrayList<>(1);
  private final AcceleratedHnswGraphOutput graphOutput;
  private final BorrowedDatasetOutput vectorOutput;
  private boolean finished;

  BorrowedDatasetHnswVectorsWriter(
      SegmentWriteState state, AcceleratedHNSWParams graphBuildParams, BulkIndexingContext context)
      throws IOException {
    this.context = context;
    this.infoStream = state.infoStream;
    AcceleratedHnswGraphOutput newGraphOutput = null;
    BorrowedDatasetOutput newVectorOutput = null;
    boolean success = false;
    try {
      newGraphOutput = new AcceleratedHnswGraphOutput(state, graphBuildParams, context.metrics());
      newVectorOutput =
          switch (context.storage()) {
            case MAPPED_SELF_CONTAINED -> new MappedSelfContainedOutput(state, context.metrics());
            case IMMUTABLE_EXTERNAL -> new ImmutableExternalFbinOutput(state, context);
            case NATIVE_BUFFERED ->
                throw new IllegalArgumentException("Borrowed writer requires a mapped dataset");
          };
      success = true;
    } finally {
      graphOutput = newGraphOutput;
      vectorOutput = newVectorOutput;
      if (!success) {
        IOUtils.closeWhileHandlingException(newVectorOutput, newGraphOutput);
      }
    }
    printInfoStream(infoStream, getClass().getSimpleName(), "borrowed FBIN writer initialized");
  }

  @Override
  public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
    if (!fields.isEmpty()) {
      throw new UnsupportedOperationException(
          "Mapped FBIN bulk builds support exactly one vector field");
    }
    if (fieldInfo.getVectorDimension() != context.dataset().dimensions()) {
      throw new IllegalArgumentException(
          "Vector field dimension "
              + fieldInfo.getVectorDimension()
              + " does not match mapped FBIN dimension "
              + context.dataset().dimensions());
    }
    BorrowedDatasetFieldWriter field =
        new BorrowedDatasetFieldWriter(fieldInfo, context.exactVectorCount());
    fields.add(field);
    return field;
  }

  @Override
  public void flush(int maxDoc, DocMap sortMap) throws IOException {
    if (sortMap != null) {
      throw new UnsupportedOperationException("Mapped FBIN bulk builds do not support index sort");
    }
    if (fields.size() != 1) {
      throw new IllegalStateException(
          "Mapped FBIN bulk builds require exactly one vector field, got " + fields.size());
    }
    BorrowedDatasetFieldWriter field = fields.getFirst();
    if (field.count() != context.exactVectorCount()) {
      throw new IllegalStateException(
          "Expected "
              + context.exactVectorCount()
              + " mapped FBIN vectors but received "
              + field.count());
    }
    if (maxDoc != context.exactVectorCount()) {
      throw new IllegalStateException(
          "Mapped FBIN bulk builds require one vector per document; maxDoc="
              + maxDoc
              + ", expected="
              + context.exactVectorCount());
    }
    vectorOutput.writeField(
        field.fieldInfo(), context.dataset(), maxDoc, field.docsWithField(), graphOutput);
  }

  @Override
  public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) {
    throw new UnsupportedOperationException("Mapped FBIN bulk builds do not support merges");
  }

  @Override
  public void finish() throws IOException {
    if (finished) {
      throw new IllegalStateException("already finished");
    }
    finished = true;
    vectorOutput.finish();
    graphOutput.finish();
  }

  @Override
  public void close() throws IOException {
    printInfoStream(infoStream, getClass().getSimpleName(), "closing resources");
    try {
      IOUtils.close(vectorOutput, graphOutput);
    } finally {
      closeCuVSResourcesInstance();
    }
  }

  @Override
  public long ramBytesUsed() {
    long total = SHALLOW_SIZE;
    for (BorrowedDatasetFieldWriter field : fields) {
      total += field.ramBytesUsed();
    }
    return total;
  }
}

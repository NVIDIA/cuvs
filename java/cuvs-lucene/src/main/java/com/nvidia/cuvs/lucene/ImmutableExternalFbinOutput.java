/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentWriteState;

/** Persists an FBIN descriptor while validation overlaps GPU graph construction. */
final class ImmutableExternalFbinOutput implements BorrowedDatasetOutput {

  private final ExternalFbinReferenceWriter referenceOutput;
  private final ExternalFbinReference reference;
  private final ExternalFbinOptions options;
  private final CagraHnswBuildMetrics metrics;

  ImmutableExternalFbinOutput(SegmentWriteState state, BulkIndexingContext context)
      throws IOException {
    this.referenceOutput = new ExternalFbinReferenceWriter(state);
    this.reference = context.reference();
    this.options = context.externalOptions();
    this.metrics = context.metrics();
  }

  @Override
  public void writeField(
      FieldInfo field,
      ExternalFloat32Dataset dataset,
      int maxDoc,
      DocsWithFieldSet docsWithField,
      AcceleratedHnswGraphOutput graphOutput)
      throws IOException {
    if (docsWithField.cardinality() != maxDoc) {
      throw new IllegalStateException(
          "Immutable external FBIN requires one vector per document; maxDoc="
              + maxDoc
              + ", vectors="
              + docsWithField.cardinality());
    }
    referenceOutput.writeField(field, reference);
    if (options.validation() == ExternalFbinBuildValidation.TRUSTED_IMMUTABLE) {
      graphOutput.writeBorrowedField(field, dataset.matrix());
      return;
    }

    long overlapStartedAt = CagraHnswBuildMetrics.start();
    try {
      ExternalFbinScanCoordinator coordinator =
          ExternalFbinScanCoordinator.start(
              reference, options.validation(), options.scanHeadStartBytes(), metrics);
      coordinator.runAfterHeadStart(() -> graphOutput.writeBorrowedField(field, dataset.matrix()));
    } finally {
      metrics.stop("external reference overlap wall [GPU+DISK]", overlapStartedAt);
    }
  }

  @Override
  public void finish() throws IOException {
    referenceOutput.finish();
  }

  @Override
  public void close() throws IOException {
    referenceOutput.close();
  }
}

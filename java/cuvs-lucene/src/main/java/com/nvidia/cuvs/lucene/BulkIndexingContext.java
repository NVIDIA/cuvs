/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.util.Objects;

/** Package-private build state reachable only through {@link CagraHnswBulkIndexWriter}. */
final class BulkIndexingContext {

  enum Storage {
    NATIVE_BUFFERED,
    MAPPED_SELF_CONTAINED,
    IMMUTABLE_EXTERNAL
  }

  private final int exactVectorCount;
  private final Storage storage;
  private final ExternalFloat32Dataset dataset;
  private final ExternalFbinReference reference;
  private final ExternalFbinOptions externalOptions;
  private final CagraHnswBuildMetrics metrics;

  static BulkIndexingContext nativeBuffered(int exactVectorCount, CagraHnswBuildMetrics metrics) {
    return new BulkIndexingContext(
        exactVectorCount, Storage.NATIVE_BUFFERED, null, null, null, metrics);
  }

  static BulkIndexingContext mapped(ExternalFloat32Dataset dataset, CagraHnswBuildMetrics metrics) {
    Objects.requireNonNull(dataset, "dataset");
    return new BulkIndexingContext(
        dataset.rows(), Storage.MAPPED_SELF_CONTAINED, dataset, null, null, metrics);
  }

  static BulkIndexingContext external(
      ImmutableExternalFbinDataset dataset,
      ExternalFbinOptions options,
      CagraHnswBuildMetrics metrics) {
    Objects.requireNonNull(dataset, "dataset");
    Objects.requireNonNull(options, "options");
    ExternalFbinReference reference = dataset.reference();
    if (options.scanHeadStartBytes() > reference.payloadLength()) {
      throw new IllegalArgumentException(
          "scanHeadStartBytes exceeds the referenced payload length: "
              + options.scanHeadStartBytes()
              + " > "
              + reference.payloadLength());
    }
    return new BulkIndexingContext(
        dataset.rows(), Storage.IMMUTABLE_EXTERNAL, dataset.dataset(), reference, options, metrics);
  }

  private BulkIndexingContext(
      int exactVectorCount,
      Storage storage,
      ExternalFloat32Dataset dataset,
      ExternalFbinReference reference,
      ExternalFbinOptions externalOptions,
      CagraHnswBuildMetrics metrics) {
    if (exactVectorCount <= 0) {
      throw new IllegalArgumentException("exactVectorCount must be positive");
    }
    this.exactVectorCount = exactVectorCount;
    this.storage = Objects.requireNonNull(storage, "storage");
    this.dataset = dataset;
    this.reference = reference;
    this.externalOptions = externalOptions;
    this.metrics = Objects.requireNonNull(metrics, "metrics");
  }

  int exactVectorCount() {
    return exactVectorCount;
  }

  Storage storage() {
    return storage;
  }

  ExternalFloat32Dataset dataset() {
    return dataset;
  }

  ExternalFbinReference reference() {
    return reference;
  }

  ExternalFbinOptions externalOptions() {
    return externalOptions;
  }

  CagraHnswBuildMetrics metrics() {
    return metrics;
  }
}

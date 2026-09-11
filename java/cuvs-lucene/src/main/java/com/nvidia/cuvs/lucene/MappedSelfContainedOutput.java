/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.io.InterruptedIOException;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentWriteState;

/** Writes a normal Lucene flat-vector file while graph construction consumes the same mapping. */
final class MappedSelfContainedOutput implements BorrowedDatasetOutput {

  private final NativeFlatVectorsWriter flatOutput;
  private final CagraHnswBuildMetrics metrics;

  MappedSelfContainedOutput(SegmentWriteState state, CagraHnswBuildMetrics metrics)
      throws IOException {
    this.flatOutput = new NativeFlatVectorsWriter(state, metrics);
    this.metrics = metrics;
  }

  @Override
  public void writeField(
      FieldInfo field,
      ExternalFloat32Dataset dataset,
      int maxDoc,
      DocsWithFieldSet docsWithField,
      AcceleratedHnswGraphOutput graphOutput)
      throws IOException {
    long overlapStartedAt = CagraHnswBuildMetrics.start();
    try (ExecutorService executor =
        Executors.newSingleThreadExecutor(
            task -> Thread.ofPlatform().name("cuvs-mapped-flat-writer").unstarted(task))) {
      Future<?> flatWrite =
          executor.submit(
              () -> {
                long startedAt = CagraHnswBuildMetrics.start();
                flatOutput.writeField(
                    field, dataset.matrix(), dataset.memorySegment(), maxDoc, docsWithField);
                metrics.stop(
                    "mapped flat-write worker [DISK]",
                    startedAt,
                    dataset.memorySegment().byteSize());
                return null;
              });

      Throwable failure = null;
      try {
        graphOutput.writeBorrowedField(field, dataset.matrix());
      } catch (Throwable graphFailure) {
        failure = graphFailure;
      }
      failure = combine(failure, await(flatWrite));
      if (failure != null) {
        throw Utils.handleThrowable(failure);
      }
    } finally {
      metrics.stop("mapped flush overlap wall [GPU+DISK]", overlapStartedAt);
    }
  }

  private static Throwable await(Future<?> future) {
    boolean interrupted = false;
    Throwable failure = null;
    while (true) {
      try {
        future.get();
        break;
      } catch (InterruptedException e) {
        interrupted = true;
      } catch (ExecutionException e) {
        failure = e.getCause();
        break;
      } catch (CancellationException e) {
        failure = e;
        break;
      }
    }
    if (interrupted) {
      Thread.currentThread().interrupt();
      InterruptedIOException interruption =
          new InterruptedIOException("Interrupted while awaiting mapped flat-vector output");
      if (failure != null) {
        interruption.addSuppressed(failure);
      }
      return interruption;
    }
    return failure;
  }

  private static Throwable combine(Throwable primary, Throwable additional) {
    if (primary == null) {
      return additional;
    }
    if (additional != null && additional != primary) {
      primary.addSuppressed(additional);
    }
    return primary;
  }

  @Override
  public void finish() throws IOException {
    flatOutput.finish();
  }

  @Override
  public void close() throws IOException {
    flatOutput.close();
  }
}

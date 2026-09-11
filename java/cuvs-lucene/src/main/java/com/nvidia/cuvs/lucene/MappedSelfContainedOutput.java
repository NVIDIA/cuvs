/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.io.InterruptedIOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
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
    try {
      runOverlapped(
          () -> {
            long startedAt = CagraHnswBuildMetrics.start();
            flatOutput.writeField(
                field, dataset.matrix(), dataset.memorySegment(), maxDoc, docsWithField);
            metrics.stop(
                "mapped flat-write worker [DISK]", startedAt, dataset.memorySegment().byteSize());
          },
          () -> graphOutput.writeBorrowedField(field, dataset.matrix()));
    } finally {
      metrics.stop("mapped flush overlap wall [GPU+DISK]", overlapStartedAt);
    }
  }

  /**
   * Runs the flat-vector write on its worker while the caller builds the graph. Package visibility
   * keeps the failure and cancellation protocol independently testable without native resources.
   */
  static void runOverlapped(OverlapTask flatWrite, OverlapTask graphWrite) throws IOException {
    ExecutorService executor =
        Executors.newSingleThreadExecutor(
            task -> Thread.ofPlatform().name("cuvs-mapped-flat-writer").unstarted(task));
    AtomicReference<Throwable> flatFailure = new AtomicReference<>();
    Future<?> flatFuture = null;
    Throwable failure = null;
    boolean interrupted = false;

    try {
      flatFuture =
          executor.submit(
              () -> {
                try {
                  flatWrite.run();
                } catch (Throwable thrown) {
                  // A cancelled Future can report completion before its running task has stopped
                  // and no longer exposes an exception thrown after cancellation. Retain the
                  // worker's actual failure until executor termination establishes visibility.
                  flatFailure.compareAndSet(null, thrown);
                }
              });

      if (Thread.interrupted()) {
        interrupted = true;
        failure = interruption("Interrupted before mapped graph output", null);
      } else {
        try {
          graphWrite.run();
        } catch (InterruptedException graphInterruption) {
          interrupted = true;
          failure =
              interruption("Interrupted while writing mapped graph output", graphInterruption);
        } catch (Throwable graphFailure) {
          failure = graphFailure;
        }
      }

      if (Thread.interrupted()) {
        interrupted = true;
        failure =
            combine(failure, interruption("Interrupted while writing mapped graph output", null));
      }
    } catch (Throwable orchestrationFailure) {
      failure = combine(failure, orchestrationFailure);
    } finally {
      if (failure != null || interrupted) {
        if (flatFuture != null) {
          flatFuture.cancel(true);
        }
        executor.shutdownNow();
      } else {
        executor.shutdown();
      }

      // Future.cancel(true) only changes Future state; it does not prove the task honored the
      // interrupt. Keep all borrowed mappings and outputs alive until the worker has really exited.
      while (executor.isTerminated() == false) {
        try {
          executor.awaitTermination(1L, TimeUnit.DAYS);
        } catch (InterruptedException awaitInterruption) {
          interrupted = true;
          failure =
              combine(
                  failure,
                  interruption(
                      "Interrupted while awaiting mapped flat-vector output", awaitInterruption));
          if (flatFuture != null) {
            flatFuture.cancel(true);
          }
          executor.shutdownNow();
        }
      }

      // Cover an interrupt arriving after the final await returned without clearing it early.
      if (Thread.interrupted()) {
        interrupted = true;
        failure =
            combine(
                failure,
                interruption("Interrupted while awaiting mapped flat-vector output", null));
      }
      failure = combine(failure, flatFailure.get());
      if (interrupted) {
        Thread.currentThread().interrupt();
      }
    }

    if (failure != null) {
      throw Utils.handleThrowable(failure);
    }
  }

  private static InterruptedIOException interruption(String message, Throwable cause) {
    InterruptedIOException interruption = new InterruptedIOException(message);
    if (cause != null) {
      interruption.initCause(cause);
    }
    return interruption;
  }

  @FunctionalInterface
  interface OverlapTask {
    void run() throws Throwable;
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

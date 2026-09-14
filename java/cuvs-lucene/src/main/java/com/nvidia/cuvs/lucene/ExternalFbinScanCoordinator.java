/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.io.InterruptedIOException;
import java.util.Objects;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.LongConsumer;

/** Coordinates a bounded validation lead followed by concurrent FBIN scanning and graph build. */
final class ExternalFbinScanCoordinator {

  @FunctionalInterface
  interface ScanAction {
    long scan(LongConsumer progress) throws IOException;
  }

  @FunctionalInterface
  interface BuildAction {
    void run() throws Throwable;
  }

  private final long requestedLeadBytes;
  private final long requiredScanBytes;
  private final long totalScanBytes;
  private final String scanStage;
  private final CagraHnswBuildMetrics metrics;
  private final AtomicLong scannedBytes = new AtomicLong();
  private final CompletableFuture<Long> leadReached = new CompletableFuture<>();
  private final ExecutorService executor;
  private final Future<Long> completion;
  private final AtomicBoolean started = new AtomicBoolean();

  static ExternalFbinScanCoordinator start(
      ExternalFbinReference reference,
      ExternalFbinBuildValidation validation,
      long requestedLeadBytes,
      CagraHnswBuildMetrics metrics) {
    Objects.requireNonNull(reference, "reference");
    Objects.requireNonNull(validation, "validation");
    if (validation == ExternalFbinBuildValidation.TRUSTED_IMMUTABLE) {
      throw new IllegalArgumentException("TRUSTED_IMMUTABLE has no external FBIN scan");
    }

    long totalScanBytes =
        validation == ExternalFbinBuildValidation.VERIFY_SHA256
            ? reference.fileLength()
            : reference.payloadLength();
    // SHA-256 must consume the complete file in order. For a nonzero first row, account for the
    // prefix before the selected payload so the requested number of selected bytes is actually hot.
    long requiredScanBytes =
        requestedLeadBytes == 0L
            ? 0L
            : validation == ExternalFbinBuildValidation.VERIFY_SHA256
                ? Math.addExact(reference.payloadOffset(), requestedLeadBytes)
                : requestedLeadBytes;
    String scanStage =
        validation == ExternalFbinBuildValidation.VERIFY_SHA256
            ? "external fbin SHA-256 [DISK+CPU]"
            : "external fbin prefetch [DISK]";
    ScanAction scanAction =
        validation == ExternalFbinBuildValidation.VERIFY_SHA256
            ? progress -> ExternalFbinIO.verifySha256(reference, progress)
            : progress -> ExternalFbinIO.prefetch(reference, progress);
    return new ExternalFbinScanCoordinator(
        requestedLeadBytes, requiredScanBytes, totalScanBytes, scanStage, scanAction, metrics);
  }

  static ExternalFbinScanCoordinator createForTests(
      long requestedLeadBytes, long requiredScanBytes, long totalScanBytes, ScanAction scanAction) {
    return new ExternalFbinScanCoordinator(
        requestedLeadBytes,
        requiredScanBytes,
        totalScanBytes,
        null,
        scanAction,
        new CagraHnswBuildMetrics());
  }

  private ExternalFbinScanCoordinator(
      long requestedLeadBytes,
      long requiredScanBytes,
      long totalScanBytes,
      String scanStage,
      ScanAction scanAction,
      CagraHnswBuildMetrics metrics) {
    if (requestedLeadBytes < 0L
        || requiredScanBytes < requestedLeadBytes
        || requiredScanBytes > totalScanBytes
        || totalScanBytes <= 0L) {
      throw new IllegalArgumentException(
          "Invalid external FBIN scan bounds: requested="
              + requestedLeadBytes
              + ", required="
              + requiredScanBytes
              + ", total="
              + totalScanBytes);
    }
    this.requestedLeadBytes = requestedLeadBytes;
    this.requiredScanBytes = requiredScanBytes;
    this.totalScanBytes = totalScanBytes;
    this.scanStage = scanStage;
    this.metrics = Objects.requireNonNull(metrics, "metrics");
    if (requiredScanBytes == 0L) {
      leadReached.complete(0L);
    }

    executor =
        Executors.newSingleThreadExecutor(
            task -> {
              Thread thread = new Thread(task, "cuvs-external-fbin-validator");
              thread.setDaemon(false);
              return thread;
            });
    completion = executor.submit(() -> scan(Objects.requireNonNull(scanAction, "scanAction")));
  }

  private long scan(ScanAction scanAction) throws IOException {
    long start = CagraHnswBuildMetrics.start();
    try {
      long scanned = scanAction.scan(this::recordProgress);
      if (scanned != totalScanBytes) {
        throw new IOException(
            "External FBIN scan reported " + scanned + " bytes; expected " + totalScanBytes);
      }
      recordProgress(scanned);
      // Do not release a full-scan lead until final verification (including digest comparison) has
      // succeeded. Partial leads intentionally allow a later validation failure to abort the build.
      leadReached.complete(scanned);
      return scanned;
    } catch (IOException | RuntimeException | Error failure) {
      leadReached.completeExceptionally(failure);
      throw failure;
    } finally {
      if (scanStage != null) {
        metrics.stop(scanStage, start, scannedBytes.get());
      }
    }
  }

  private void recordProgress(long current) {
    long previous = scannedBytes.get();
    if (current < previous || current > totalScanBytes) {
      throw new IllegalStateException(
          "External FBIN scan progress is invalid: previous="
              + previous
              + ", current="
              + current
              + ", total="
              + totalScanBytes);
    }
    scannedBytes.set(current);
    if (requiredScanBytes != totalScanBytes && current >= requiredScanBytes) {
      leadReached.complete(current);
    }
  }

  void runAfterHeadStart(BuildAction buildAction) throws IOException {
    Objects.requireNonNull(buildAction, "buildAction");
    if (!started.compareAndSet(false, true)) {
      throw new IllegalStateException("External FBIN scan coordinator has already been used");
    }

    Throwable failure = null;
    boolean interrupted = false;
    boolean buildStarted = false;
    long postLeadStart = 0L;
    try {
      long headStartTimer = CagraHnswBuildMetrics.start();
      try {
        leadReached.get();
      } catch (InterruptedException e) {
        interrupted = true;
        failure = interruptedFailure("Interrupted while awaiting external FBIN scan head start", e);
      } catch (ExecutionException e) {
        failure = e.getCause();
      } finally {
        if (requestedLeadBytes != 0L) {
          metrics.stop("external fbin scan head-start [DISK]", headStartTimer, scannedBytes.get());
        }
      }

      if (failure == null && Thread.interrupted()) {
        interrupted = true;
        failure =
            interruptedFailure("Interrupted before building with an external FBIN scan", null);
      }

      if (failure == null) {
        buildStarted = true;
        if (requestedLeadBytes != 0L) {
          metrics.addCounter("external fbin requested head-start bytes", requestedLeadBytes);
          metrics.addCounter("external fbin required scan bytes", requiredScanBytes);
          metrics.addCounter("external fbin scan bytes at cagra start", scannedBytes.get());
          postLeadStart = CagraHnswBuildMetrics.start();
        }
        try {
          buildAction.run();
        } catch (Throwable buildFailure) {
          failure = buildFailure;
        } finally {
          if (requestedLeadBytes != 0L) {
            metrics.addCounter("external fbin scan bytes at cagra end", scannedBytes.get());
          }
        }
      }

      if (Thread.interrupted()) {
        interrupted = true;
        failure =
            combine(
                failure,
                interruptedFailure("Interrupted while building with an external FBIN scan", null));
      }
      if (failure != null) {
        completion.cancel(true);
      }

      try {
        completion.get();
      } catch (InterruptedException e) {
        interrupted = true;
        completion.cancel(true);
        failure =
            combine(
                failure,
                interruptedFailure("Interrupted while awaiting external FBIN validation", e));
      } catch (ExecutionException e) {
        failure = combine(failure, e.getCause());
      } catch (CancellationException cancellation) {
        if (failure == null) {
          failure = cancellation;
        }
      }
    } finally {
      if (failure != null) {
        completion.cancel(true);
        executor.shutdownNow();
      } else {
        executor.shutdown();
      }
      executor.close();
      if (requestedLeadBytes != 0L && buildStarted) {
        metrics.stop("external fbin post-lead overlap wall [GPU+DISK]", postLeadStart);
      }
      if (interrupted) {
        Thread.currentThread().interrupt();
      }
    }

    if (failure != null) {
      Utils.handleThrowable(failure);
    }
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

  private static InterruptedIOException interruptedFailure(String message, Throwable cause) {
    InterruptedIOException failure = new InterruptedIOException(message);
    if (cause != null) {
      failure.initCause(cause);
    }
    return failure;
  }
}

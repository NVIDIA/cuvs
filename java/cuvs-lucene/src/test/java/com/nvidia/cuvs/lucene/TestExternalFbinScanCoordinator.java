/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.io.InterruptedIOException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.apache.lucene.tests.util.LuceneTestCase;

public class TestExternalFbinScanCoordinator extends LuceneTestCase {

  public void testZeroLeadStartsBuildWithoutWaitingForScanProgress() throws Exception {
    CountDownLatch buildStarted = new CountDownLatch(1);
    ExternalFbinScanCoordinator coordinator =
        ExternalFbinScanCoordinator.createForTests(
            0L,
            0L,
            10L,
            progress -> {
              await(buildStarted);
              progress.accept(10L);
              return 10L;
            });

    coordinator.runAfterHeadStart(buildStarted::countDown);
  }

  public void testZeroLeadPreInterruptedCallerDoesNotStartBuild() throws Exception {
    CountDownLatch releaseScan = new CountDownLatch(1);
    AtomicBoolean buildStarted = new AtomicBoolean();
    ExternalFbinScanCoordinator coordinator =
        ExternalFbinScanCoordinator.createForTests(
            0L,
            0L,
            16L,
            progress -> {
              try {
                releaseScan.await();
              } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("scan interrupted", e);
              }
              progress.accept(16L);
              return 16L;
            });

    Thread.currentThread().interrupt();
    try {
      expectThrows(
          InterruptedIOException.class,
          () -> coordinator.runAfterHeadStart(() -> buildStarted.set(true)));
      assertFalse(buildStarted.get());
      assertTrue(Thread.currentThread().isInterrupted());
    } finally {
      releaseScan.countDown();
      Thread.interrupted();
    }
  }

  public void testBuildStartsOnlyAfterRequiredProgress() throws Exception {
    AtomicBoolean buildStarted = new AtomicBoolean();
    CountDownLatch releaseScanner = new CountDownLatch(1);
    ExternalFbinScanCoordinator coordinator =
        ExternalFbinScanCoordinator.createForTests(
            8L,
            8L,
            10L,
            progress -> {
              progress.accept(4L);
              assertFalse(buildStarted.get());
              progress.accept(8L);
              await(releaseScanner);
              progress.accept(10L);
              return 10L;
            });

    coordinator.runAfterHeadStart(
        () -> {
          buildStarted.set(true);
          releaseScanner.countDown();
        });
    assertTrue(buildStarted.get());
  }

  public void testFailureBeforeLeadPreventsBuild() throws Exception {
    AtomicBoolean buildStarted = new AtomicBoolean();
    ExternalFbinScanCoordinator coordinator =
        ExternalFbinScanCoordinator.createForTests(
            8L,
            8L,
            10L,
            progress -> {
              progress.accept(4L);
              throw new IOException("scan failed before lead");
            });

    IOException failure =
        expectThrows(
            IOException.class, () -> coordinator.runAfterHeadStart(() -> buildStarted.set(true)));
    assertEquals("scan failed before lead", failure.getMessage());
    assertFalse(buildStarted.get());
  }

  public void testFailureAfterLeadFailsCompletedBuild() throws Exception {
    CountDownLatch buildStarted = new CountDownLatch(1);
    ExternalFbinScanCoordinator coordinator =
        ExternalFbinScanCoordinator.createForTests(
            8L,
            8L,
            10L,
            progress -> {
              progress.accept(8L);
              await(buildStarted);
              throw new IOException("scan failed after lead");
            });

    IOException failure =
        expectThrows(
            IOException.class, () -> coordinator.runAfterHeadStart(buildStarted::countDown));
    assertEquals("scan failed after lead", failure.getMessage());
    assertEquals(0L, buildStarted.getCount());
  }

  public void testBuildFailureCancelsScanner() throws Exception {
    CountDownLatch scannerWaiting = new CountDownLatch(1);
    AtomicBoolean scannerInterrupted = new AtomicBoolean();
    ExternalFbinScanCoordinator coordinator =
        ExternalFbinScanCoordinator.createForTests(
            1L,
            1L,
            10L,
            progress -> {
              progress.accept(1L);
              scannerWaiting.countDown();
              try {
                new CountDownLatch(1).await();
                throw new AssertionError("Scanner was not cancelled");
              } catch (InterruptedException expected) {
                scannerInterrupted.set(true);
                throw new IOException("scanner cancelled", expected);
              }
            });

    IOException failure =
        expectThrows(
            IOException.class,
            () ->
                coordinator.runAfterHeadStart(
                    () -> {
                      assertTrue(scannerWaiting.await(10, TimeUnit.SECONDS));
                      throw new IOException("graph failed");
                    }));
    assertEquals("graph failed", failure.getMessage());
    assertTrue(scannerInterrupted.get());
  }

  public void testCallerInterruptionCancelsScanAndPreservesInterrupt() throws Exception {
    CountDownLatch scannerStarted = new CountDownLatch(1);
    AtomicBoolean scannerInterrupted = new AtomicBoolean();
    ExternalFbinScanCoordinator coordinator =
        ExternalFbinScanCoordinator.createForTests(
            1L,
            1L,
            10L,
            progress -> {
              scannerStarted.countDown();
              try {
                new CountDownLatch(1).await();
                throw new AssertionError("Scanner was not cancelled");
              } catch (InterruptedException expected) {
                scannerInterrupted.set(true);
                throw new IOException("scanner cancelled", expected);
              }
            });

    assertTrue(scannerStarted.await(10, TimeUnit.SECONDS));
    Thread.currentThread().interrupt();
    try {
      expectThrows(
          InterruptedIOException.class, () -> coordinator.runAfterHeadStart(() -> fail("build")));
      assertTrue(Thread.currentThread().isInterrupted());
      assertTrue(scannerInterrupted.get());
    } finally {
      Thread.interrupted();
    }
  }

  private static void await(CountDownLatch latch) throws IOException {
    try {
      assertTrue(latch.await(10, TimeUnit.SECONDS));
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IOException("Interrupted while awaiting test coordination", e);
    }
  }
}

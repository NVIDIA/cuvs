/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.io.InterruptedIOException;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.apache.lucene.store.DataOutput;
import org.apache.lucene.tests.util.LuceneTestCase;

public class TestMappedSelfContainedOutputCancellation extends LuceneTestCase {

  public void testGraphFailureCancelsAndWaitsForFlatWorker() throws Exception {
    CountDownLatch flatStarted = new CountDownLatch(1);
    CountDownLatch flatInterrupted = new CountDownLatch(1);
    CountDownLatch releaseFlat = new CountDownLatch(1);
    AtomicBoolean flatExited = new AtomicBoolean();
    AtomicBoolean returnedAfterFlatExit = new AtomicBoolean();
    AtomicReference<Throwable> observed = new AtomicReference<>();
    IOException graphFailure = new IOException("graph failed");
    IOException flatFailure = new IOException("flat worker failed after cancellation");

    Thread caller =
        Thread.ofPlatform()
            .name("mapped-overlap-graph-failure-test")
            .unstarted(
                () -> {
                  try {
                    MappedSelfContainedOutput.runOverlapped(
                        () -> {
                          flatStarted.countDown();
                          try {
                            new CountDownLatch(1).await();
                            throw new AssertionError("flat worker unexpectedly resumed");
                          } catch (InterruptedException expected) {
                            flatInterrupted.countDown();
                            awaitUninterruptibly(releaseFlat);
                            throw flatFailure;
                          } finally {
                            flatExited.set(true);
                          }
                        },
                        () -> {
                          assertTrue(flatStarted.await(10L, TimeUnit.SECONDS));
                          throw graphFailure;
                        });
                    observed.set(new AssertionError("expected graph failure"));
                  } catch (Throwable thrown) {
                    observed.set(thrown);
                  } finally {
                    returnedAfterFlatExit.set(flatExited.get());
                  }
                });

    caller.start();
    try {
      assertTrue(flatInterrupted.await(10L, TimeUnit.SECONDS));
      assertFalse(flatExited.get());
    } finally {
      releaseFlat.countDown();
      caller.join(10_000L);
      if (caller.isAlive()) {
        caller.interrupt();
      }
    }

    assertFalse("overlap caller did not terminate", caller.isAlive());
    assertSame(graphFailure, observed.get());
    assertTrue(flatExited.get());
    assertTrue("overlap returned before the flat worker exited", returnedAfterFlatExit.get());
    assertArrayEquals(new Throwable[] {flatFailure}, graphFailure.getSuppressed());
  }

  public void testCallerInterruptionCancelsWorkerAndRestoresInterrupt() throws Exception {
    CountDownLatch flatStarted = new CountDownLatch(1);
    CountDownLatch graphFinished = new CountDownLatch(1);
    CountDownLatch releaseFlat = new CountDownLatch(1);
    AtomicBoolean flatInterrupted = new AtomicBoolean();
    AtomicBoolean flatExited = new AtomicBoolean();
    AtomicBoolean returnedAfterFlatExit = new AtomicBoolean();
    AtomicBoolean interruptRestored = new AtomicBoolean();
    AtomicReference<Throwable> observed = new AtomicReference<>();

    Thread caller =
        Thread.ofPlatform()
            .name("mapped-overlap-caller-interruption-test")
            .unstarted(
                () -> {
                  try {
                    MappedSelfContainedOutput.runOverlapped(
                        () -> {
                          flatStarted.countDown();
                          try {
                            releaseFlat.await();
                          } catch (InterruptedException expected) {
                            flatInterrupted.set(true);
                          } finally {
                            flatExited.set(true);
                          }
                        },
                        () -> {
                          assertTrue(flatStarted.await(10L, TimeUnit.SECONDS));
                          graphFinished.countDown();
                        });
                    observed.set(new AssertionError("expected caller interruption"));
                  } catch (Throwable thrown) {
                    observed.set(thrown);
                    interruptRestored.set(Thread.currentThread().isInterrupted());
                  } finally {
                    returnedAfterFlatExit.set(flatExited.get());
                  }
                });

    caller.start();
    try {
      assertTrue(graphFinished.await(10L, TimeUnit.SECONDS));
      caller.interrupt();
      caller.join(10_000L);
    } finally {
      releaseFlat.countDown();
      if (caller.isAlive()) {
        caller.interrupt();
      }
    }

    assertFalse("interrupted overlap caller did not terminate", caller.isAlive());
    assertTrue(observed.get() instanceof InterruptedIOException);
    assertTrue(flatInterrupted.get());
    assertTrue(flatExited.get());
    assertTrue("overlap returned before the flat worker exited", returnedAfterFlatExit.get());
    assertTrue("caller interrupt status was not restored", interruptRestored.get());
  }

  public void testRawCopyChecksInterruptionBetweenChunks() throws Exception {
    AtomicInteger writes = new AtomicInteger();
    DataOutput interruptingOutput =
        new DataOutput() {
          @Override
          public void writeByte(byte value) {
            throw new AssertionError("unexpected single-byte write");
          }

          @Override
          public void writeBytes(byte[] bytes, int offset, int length) {
            if (writes.incrementAndGet() == 1) {
              Thread.currentThread().interrupt();
            }
          }
        };

    try {
      InterruptedIOException failure =
          expectThrows(
              InterruptedIOException.class,
              () ->
                  NativeFlatVectorsWriter.copyRawFloat32Vectors(
                      MemorySegment.ofArray(new byte[8]),
                      8L,
                      interruptingOutput,
                      new CagraHnswBuildMetrics(),
                      4));
      assertTrue(failure.getMessage(), failure.getMessage().contains("at byte 4 of 8"));
      assertEquals(1, writes.get());
      assertTrue(Thread.currentThread().isInterrupted());
    } finally {
      Thread.interrupted();
    }
  }

  private static void awaitUninterruptibly(CountDownLatch latch) {
    boolean interrupted = false;
    while (latch.getCount() != 0L) {
      try {
        latch.await();
      } catch (InterruptedException expected) {
        interrupted = true;
      }
    }
    if (interrupted) {
      Thread.currentThread().interrupt();
    }
  }
}

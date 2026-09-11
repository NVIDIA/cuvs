/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

public class TestCagraHnswBuildMetricsConcurrency extends LuceneTestCase {

  @Test
  public void testGaugeAllowsRepeatedIdenticalValues() {
    CagraHnswBuildMetrics metrics = new CagraHnswBuildMetrics();

    metrics.setGauge("effective graph degree", 32L);
    metrics.setGauge("effective graph degree", 32L);

    assertEquals(32L, metrics.snapshot().get("gauge/effective graph degree").longValue());
  }

  @Test
  public void testGaugeRejectsConflictingValues() {
    CagraHnswBuildMetrics metrics = new CagraHnswBuildMetrics();
    metrics.setGauge("effective graph degree", 32L);

    IllegalStateException failure =
        expectThrows(
            IllegalStateException.class, () -> metrics.setGauge("effective graph degree", 56L));

    assertTrue(failure.getMessage(), failure.getMessage().contains("effective graph degree"));
    assertTrue(failure.getMessage(), failure.getMessage().contains("32"));
    assertTrue(failure.getMessage(), failure.getMessage().contains("56"));
    assertEquals(32L, metrics.snapshot().get("gauge/effective graph degree").longValue());
  }

  @Test
  public void testCounterRemainsAdditive() {
    CagraHnswBuildMetrics metrics = new CagraHnswBuildMetrics();

    metrics.addCounter("logical cagra adjacency bytes", 4_096L);
    metrics.addCounter("logical cagra adjacency bytes", 8_192L);

    assertEquals(
        12_288L, metrics.snapshot().get("counter/logical cagra adjacency bytes").longValue());
  }

  @Test
  public void testSnapshotWhileNewStagesAreRecorded() throws Exception {
    CagraHnswBuildMetrics metrics = new CagraHnswBuildMetrics();
    CountDownLatch start = new CountDownLatch(1);
    AtomicBoolean recording = new AtomicBoolean(true);

    try (ExecutorService executor = Executors.newFixedThreadPool(2)) {
      Future<?> recorder =
          executor.submit(
              () -> {
                await(start);
                try {
                  for (int i = 0; i < 20_000; i++) {
                    metrics.record("stage-" + i, i, i + 1L);
                  }
                } finally {
                  recording.set(false);
                }
              });
      Future<?> snapshotter =
          executor.submit(
              () -> {
                await(start);
                while (recording.get()) {
                  metrics.snapshot();
                }
              });

      start.countDown();
      recorder.get();
      snapshotter.get();
    }

    assertEquals(20_000, metrics.snapshot().size() / 3);
  }

  private static void await(CountDownLatch latch) {
    try {
      latch.await();
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      throw new AssertionError("interrupted while starting concurrency test", interrupted);
    }
  }
}

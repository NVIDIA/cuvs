/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.nvidia.cuvs.FilterBitsetHandle;
import com.nvidia.cuvs.lucene.FilterBitsetCache.CachedFilterBitset;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/**
 * Concurrency and sizing tests for {@link FilterBitsetCache}. These exercise the compute-once,
 * reference-counting, and byte-budget behavior in isolation by injecting a fake {@link
 * FilterBitsetHandle}, so no GPU or native library is required.
 */
public class TestFilterBitsetCache {

  /** A byte size comfortably larger than any single test's total entries, so nothing evicts. */
  private static final long HUGE_BUDGET = 1L << 40;

  /** Representative per-entry byte size used by tests that don't care about sizing. */
  private static final long ENTRY_BYTES = 64;

  private FilterBitsetCache cache;

  @Before
  public void resetCache() {
    cache = new FilterBitsetCache(new FilterBitsetCacheConfig(true, HUGE_BUDGET));
  }

  @After
  public void restoreDefaults() {
    cache.clear();
    cache = null;
  }

  /**
   * Fake handle faithfully modeling the {@link FilterBitsetHandle} reference-counting contract, so
   * the cache's use of it can be observed. {@link #decRef()} throws if released more times than
   * referenced, turning any over-release by the cache into a test failure.
   */
  private static final class CountingHandle implements FilterBitsetHandle {
    final AtomicInteger refCount = new AtomicInteger(1); // initial reference, as in the real handle
    final AtomicInteger closeCalls = new AtomicInteger();
    final AtomicBoolean freed = new AtomicBoolean();
    private final AtomicBoolean initialReleased = new AtomicBoolean();

    @Override
    public boolean tryIncRef() {
      int c;
      do {
        c = refCount.get();
        if (c == 0) return false;
      } while (!refCount.compareAndSet(c, c + 1));
      return true;
    }

    @Override
    public void decRef() {
      int c = refCount.decrementAndGet();
      if (c == 0) {
        freed.set(true);
      } else if (c < 0) {
        throw new IllegalStateException("decRef() called more times than references were taken");
      }
    }

    @Override
    public void close() {
      closeCalls.incrementAndGet();
      if (initialReleased.compareAndSet(false, true)) {
        decRef();
      }
    }
  }

  private static Object segKey(String s) {
    return s;
  }

  /** Wraps a handle as a cache value; most tests here don't care about the cardinality. */
  private static CachedFilterBitset cached(FilterBitsetHandle handle) {
    return new CachedFilterBitset(handle, 1);
  }

  /** Concurrent misses on the same key build the handle exactly once and each get their own ref. */
  @Test
  public void computeOnceUnderConcurrentAcquire() throws Exception {
    final int threads = 32;
    AtomicInteger buildCount = new AtomicInteger();
    CountingHandle handle = new CountingHandle();
    final String field = "computeOnce";

    ExecutorService pool = Executors.newFixedThreadPool(threads);
    CountDownLatch start = new CountDownLatch(1);
    List<Future<CachedFilterBitset>> results = new ArrayList<>();
    try {
      for (int i = 0; i < threads; i++) {
        results.add(
            pool.submit(
                () -> {
                  start.await();
                  return cache.acquire(
                      null,
                      segKey(field),
                      field,
                      ENTRY_BYTES,
                      () -> {
                        buildCount.incrementAndGet();
                        // Widen the race window so waiters reach the future before it completes.
                        try {
                          Thread.sleep(20);
                        } catch (InterruptedException e) {
                          Thread.currentThread().interrupt();
                        }
                        return cached(handle);
                      });
                }));
      }
      start.countDown();
      for (Future<CachedFilterBitset> r : results) {
        assertSame(handle, r.get(10, TimeUnit.SECONDS).handle());
      }
    } finally {
      pool.shutdownNow();
      pool.awaitTermination(5, TimeUnit.SECONDS);
    }

    assertEquals("builder must run exactly once for a key", 1, buildCount.get());
    // One cache reference plus one caller reference per acquiring thread.
    assertEquals(1 + threads, handle.refCount.get());
    assertFalse(handle.freed.get());

    // Releasing every caller reference leaves the cache's own reference intact.
    for (int i = 0; i < threads; i++) {
      handle.decRef();
    }
    assertEquals(1, handle.refCount.get());
    assertFalse(handle.freed.get());
  }

  /** Once the byte budget is exceeded, the eldest entry is evicted and freed; newer ones stay. */
  @Test
  public void byteCapEvictsEldestWhenBudgetExceeded() throws Exception {
    // Budget holds two 64-byte entries; inserting a third must evict exactly the oldest.
    cache = new FilterBitsetCache(new FilterBitsetCacheConfig(true, 2 * ENTRY_BYTES));

    List<CountingHandle> handles = new ArrayList<>();
    for (int i = 0; i < 3; i++) {
      CountingHandle h = new CountingHandle();
      handles.add(h);
      final String field = "bcap-" + i;
      FilterBitsetHandle got =
          cache.acquire(null, segKey(field), field, ENTRY_BYTES, () -> cached(h)).handle();
      assertSame(h, got);
      got.decRef(); // caller finished; the cache keeps its reference
    }

    assertTrue("oldest handle should have been evicted and freed", handles.get(0).freed.get());
    assertEquals("evicted handle closed exactly once", 1, handles.get(0).closeCalls.get());
    assertFalse("second handle should still be cached", handles.get(1).freed.get());
    assertFalse("third handle should still be cached", handles.get(2).freed.get());
    assertTrue(
        "total bytes must stay within budget", cache.currentBytesForTests() <= 2 * ENTRY_BYTES);
  }

  /** clear() releases every cache reference (freeing all handles) and resets byte accounting. */
  @Test
  public void clearFreesAllAndResets() throws Exception {
    List<CountingHandle> handles = new ArrayList<>();
    for (int i = 0; i < 3; i++) {
      CountingHandle h = new CountingHandle();
      handles.add(h);
      final String field = "clr-" + i;
      cache.acquire(null, segKey(field), field, ENTRY_BYTES, () -> cached(h)).handle().decRef();
    }
    assertTrue("entries charged to the budget", cache.currentBytesForTests() > 0);

    cache.clear();

    for (int i = 0; i < handles.size(); i++) {
      CountingHandle h = handles.get(i);
      assertTrue("handle " + i + " freed by clear()", h.freed.get());
      assertEquals("handle " + i + " closed exactly once", 1, h.closeCalls.get());
    }
    assertEquals("byte accounting reset", 0, cache.currentBytesForTests());
  }

  /** An entry larger than the whole budget is never cached; each acquire rebuilds it uncached. */
  @Test
  public void oversizedEntryIsNotCached() throws Exception {
    cache = new FilterBitsetCache(new FilterBitsetCacheConfig(true, 2 * ENTRY_BYTES));
    final long oversized = 4 * ENTRY_BYTES; // larger than the whole budget
    final String field = "oversized";
    AtomicInteger buildCount = new AtomicInteger();

    CountingHandle h1 = new CountingHandle();
    FilterBitsetHandle got1 =
        cache
            .acquire(
                null,
                segKey(field),
                field,
                oversized,
                () -> {
                  buildCount.incrementAndGet();
                  return cached(h1);
                })
            .handle();
    assertSame(h1, got1);
    got1.decRef(); // caller-owned; nothing else holds it
    assertTrue("uncached oversized handle freed once caller releases it", h1.freed.get());
    assertEquals("nothing charged to the budget", 0, cache.currentBytesForTests());

    CountingHandle h2 = new CountingHandle();
    FilterBitsetHandle got2 =
        cache
            .acquire(
                null,
                segKey(field),
                field,
                oversized,
                () -> {
                  buildCount.incrementAndGet();
                  return cached(h2);
                })
            .handle();
    assertSame(h2, got2);
    got2.decRef();
    assertEquals("oversized entry is rebuilt every time, never cached", 2, buildCount.get());
  }

  /** The public default configuration has a deterministic 128 MiB budget. */
  @Test
  public void defaultConfiguration() {
    assertTrue(FilterBitsetCacheConfig.DEFAULT.enabled());
    assertEquals(128L * 1024 * 1024, FilterBitsetCacheConfig.DEFAULT.maxBytes());
  }

  /** Enabled caches require a real budget; zero remains valid when the cache is disabled. */
  @Test
  public void enabledConfigurationRequiresPositiveBudget() {
    try {
      new FilterBitsetCacheConfig(true, 0);
      fail("enabled cache with a zero budget should be rejected");
    } catch (IllegalArgumentException expected) {
      assertEquals("maxBytes must be positive when caching is enabled", expected.getMessage());
    }

    FilterBitsetCacheConfig disabled = new FilterBitsetCacheConfig(false, 0);
    assertFalse(disabled.enabled());
    assertEquals(0, disabled.maxBytes());
  }

  /** Entries and lifecycle operations are isolated between cache instances. */
  @Test
  public void cacheInstancesAreIndependent() throws Exception {
    FilterBitsetCache other = new FilterBitsetCache(new FilterBitsetCacheConfig(true, HUGE_BUDGET));
    CountingHandle first = new CountingHandle();
    CountingHandle second = new CountingHandle();
    Object key = segKey("shared-reader-key");

    cache.acquire(null, key, "f", ENTRY_BYTES, () -> cached(first)).handle().decRef();
    other.acquire(null, key, "f", ENTRY_BYTES, () -> cached(second)).handle().decRef();

    assertFalse(first.freed.get());
    assertFalse(second.freed.get());
    assertEquals(ENTRY_BYTES, cache.currentBytesForTests());
    assertEquals(ENTRY_BYTES, other.currentBytesForTests());

    cache.clear();

    assertTrue("clearing the first cache releases its handle", first.freed.get());
    assertFalse("clearing the first cache must not affect the second", second.freed.get());
    assertEquals(0, cache.currentBytesForTests());
    assertEquals(ENTRY_BYTES, other.currentBytesForTests());

    other.clear();
    assertTrue("the second cache releases its own handle", second.freed.get());
  }

  /** The enabled flag reflects the instance's immutable configuration. */
  @Test
  public void enabledFlagReflectsConfiguration() {
    assertTrue(cache.isEnabled());
    FilterBitsetCache disabled = new FilterBitsetCache(new FilterBitsetCacheConfig(false, 0));
    assertFalse(disabled.isEnabled());
  }

  /** invalidateReader evicts and frees exactly the given segment's entries, leaving others cached. */
  @Test
  public void invalidateReaderEvictsOnlyThatSegment() throws Exception {
    CountingHandle hA = new CountingHandle();
    CountingHandle hB = new CountingHandle();
    Object keyA = segKey("segA");
    Object keyB = segKey("segB");
    AtomicInteger buildA = new AtomicInteger();

    // Cache one entry per segment, releasing the caller ref so the cache keeps its own.
    cache
        .acquire(
            null,
            keyA,
            "f",
            ENTRY_BYTES,
            () -> {
              buildA.incrementAndGet();
              return cached(hA);
            })
        .handle()
        .decRef();
    cache.acquire(null, keyB, "f", ENTRY_BYTES, () -> cached(hB)).handle().decRef();

    // Invalidate segment A only, as its reader's close listener would.
    cache.invalidateReader(keyA);

    assertTrue("segment A handle freed on invalidation", hA.freed.get());
    assertEquals("segment A handle closed exactly once", 1, hA.closeCalls.get());
    assertFalse("segment B handle must remain cached", hB.freed.get());
    assertEquals("segment B handle must not be closed", 0, hB.closeCalls.get());

    // Segment A's entry is gone, so a later acquire rebuilds it.
    cache
        .acquire(
            null,
            keyA,
            "f",
            ENTRY_BYTES,
            () -> {
              buildA.incrementAndGet();
              return cached(new CountingHandle());
            })
        .handle()
        .decRef();
    assertEquals("invalidated segment rebuilds on next acquire", 2, buildA.get());
  }

  /** A failed build is not cached, so a later acquire rebuilds. */
  @Test
  public void failedBuildIsNotCachedAndCanRetry() throws Exception {
    final String field = "failing";
    AtomicInteger buildCount = new AtomicInteger();

    try {
      cache.acquire(
          null,
          segKey(field),
          field,
          ENTRY_BYTES,
          () -> {
            buildCount.incrementAndGet();
            throw new IOException("boom");
          });
      fail("expected IOException to propagate");
    } catch (IOException expected) {
      assertEquals("boom", expected.getMessage());
    }
    assertEquals(1, buildCount.get());
    assertEquals("failed build must not charge the budget", 0, cache.currentBytesForTests());

    CountingHandle handle = new CountingHandle();
    FilterBitsetHandle got =
        cache
            .acquire(
                null,
                segKey(field),
                field,
                ENTRY_BYTES,
                () -> {
                  buildCount.incrementAndGet();
                  return cached(handle);
                })
            .handle();
    assertSame(handle, got);
    assertEquals("failed entry must not be cached; retry rebuilds", 2, buildCount.get());
    got.decRef();
  }

  /**
   * The cardinality is part of the cached value: a hit returns the built cardinality without
   * rebuilding, which is what lets {@link GPUKnnFloatVectorQuery} decide on exact search for free.
   */
  @Test
  public void cardinalityIsCachedWithTheHandle() throws Exception {
    final String field = "cardinality";
    CountingHandle handle = new CountingHandle();
    AtomicInteger buildCount = new AtomicInteger();
    FilterBitsetCache.FilterBuilder builder =
        () -> {
          buildCount.incrementAndGet();
          return new CachedFilterBitset(handle, 42);
        };

    CachedFilterBitset first = cache.acquire(null, segKey(field), field, ENTRY_BYTES, builder);
    assertEquals(42, first.cardinality());
    first.handle().decRef();

    CachedFilterBitset second = cache.acquire(null, segKey(field), field, ENTRY_BYTES, builder);
    assertEquals("cached value carries the cardinality", 42, second.cardinality());
    assertSame(handle, second.handle());
    assertEquals("a cache hit must not rebuild", 1, buildCount.get());
    second.handle().decRef();
  }

  /**
   * Hammer acquire/decRef concurrently across several keys. Any unbalanced reference handling by the
   * cache trips {@link CountingHandle#decRef()}'s below-zero guard and fails the test. The budget is
   * left huge (see {@link #resetCache}) so no eviction happens mid-run.
   */
  @Test
  public void concurrentAcquireReleaseIsBalanced() throws Exception {
    final int keySpace = 8;
    final int threads = 16;
    final int iterationsPerThread = 500;

    ExecutorService pool = Executors.newFixedThreadPool(threads);
    CountDownLatch start = new CountDownLatch(1);
    List<Future<?>> tasks = new ArrayList<>();
    try {
      for (int t = 0; t < threads; t++) {
        final int seed = t;
        tasks.add(
            pool.submit(
                () -> {
                  start.await();
                  for (int i = 0; i < iterationsPerThread; i++) {
                    final String field = "stress-" + ((seed + i) % keySpace);
                    // A fresh handle per build; a cached key reuses whatever was built first.
                    FilterBitsetHandle h =
                        cache
                            .acquire(
                                null,
                                segKey(field),
                                field,
                                ENTRY_BYTES,
                                () -> cached(new CountingHandle()))
                            .handle();
                    h.decRef();
                  }
                  return null;
                }));
      }
      start.countDown();
      for (Future<?> f : tasks) {
        f.get(30, TimeUnit.SECONDS); // surfaces any exception thrown inside a task
      }
    } finally {
      pool.shutdownNow();
      pool.awaitTermination(5, TimeUnit.SECONDS);
    }
  }
}

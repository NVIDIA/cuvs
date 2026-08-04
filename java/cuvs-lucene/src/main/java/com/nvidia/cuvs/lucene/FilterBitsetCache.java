/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.FilterBitsetHandle;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.logging.Logger;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.search.Query;

/**
 * Shared LRU cache mapping (filter Query, single-segment reader key, field) → {@link
 * FilterBitsetHandle}. One entry per segment, so an index update only invalidates the changed
 * segments' entries rather than the whole cache.
 *
 * <p>Host-side cache holding packed bitset arrays; the device-side upload is managed inside {@link
 * FilterBitsetHandle} itself. Entries are evicted in LRU order once the total cached size exceeds
 * the configured byte budget.
 *
 * <h2>Sizing</h2>
 *
 * <p>The cache is bounded by a byte budget rather than an entry count, since per-segment bitsets
 * vary widely in size. The budget covers the host-side packed arrays; the device-side allocation
 * mirrors it one-for-one, so a single budget bounds both. The default is the fixed {@link
 * FilterBitsetCacheConfig#DEFAULT_MAX_BYTES} budget and can be overridden through {@link
 * FilterBitsetCacheConfig}. The budget is not preallocated: packed arrays are retained as entries
 * are inserted. Actual host usage can exceed the accounted payload bytes because of object
 * overhead, temporary filter construction, and handles that remain referenced by in-flight
 * searches after eviction.
 *
 * <h2>Concurrency</h2>
 *
 * <p>Values are stored as {@link CompletableFuture}s so that concurrent misses on the same key
 * build the handle exactly once: the first thread inserts an incomplete future and builds; others
 * find the future and await it (without holding the cache lock, so builds for different keys still
 * run in parallel).
 *
 * <p>Lifetime is reference-counted (see {@link FilterBitsetHandle}). A cached handle carries one
 * cache-owned reference, released via {@link FilterBitsetHandle#close()} when the entry is evicted.
 * {@link #acquire} returns a handle with an <em>additional</em> reference already taken; the caller
 * must {@link FilterBitsetHandle#decRef()} it after use. Because the free happens only when the last
 * reference is dropped, an eviction concurrent with an in-flight search cannot free a handle still
 * in use, and a replaced entry cannot leak.
 */
final class FilterBitsetCache {

  private static final Logger LOG = Logger.getLogger(FilterBitsetCache.class.getName());

  /**
   * Secondary guard on entry count so a pathological stream of tiny entries cannot grow the map
   * unbounded even when the byte budget is generous. The byte budget is the primary bound.
   */
  static final int MAX_ENTRIES_GUARD = 4096;

  /** Builds the handle for a key on a cache miss. */
  @FunctionalInterface
  interface FilterBuilder {
    FilterBitsetHandle build() throws IOException;
  }

  private record FilterCacheKey(Query filter, Object segReaderKey, String field) {
    @Override
    public boolean equals(Object o) {
      if (!(o instanceof FilterCacheKey other)) return false;
      return Objects.equals(filter, other.filter)
          && Objects.equals(segReaderKey, other.segReaderKey)
          && Objects.equals(field, other.field);
    }

    @Override
    public int hashCode() {
      return Objects.hash(filter, segReaderKey, field);
    }
  }

  /** A cached future together with the byte size charged to the budget for its entry. */
  private record CacheEntry(CompletableFuture<FilterBitsetHandle> future, long bytes) {}

  private final LinkedHashMap<FilterCacheKey, CacheEntry> cache =
      new LinkedHashMap<>(16, 0.75f, /* access-order= */ true);

  // Segment reader keys with a registered close listener, so each reader registers exactly once.
  private final Set<Object> registered = new HashSet<>();

  // ---- Configuration and accounting (guarded by this) ----

  private final boolean enabled;

  /** Fixed byte budget for this cache instance. */
  private final long maxBytes;

  private boolean workingSetWarned;
  private boolean oversizedWarned;

  /** Total bytes charged to the budget across all live cache entries. */
  private long totalBytes;

  FilterBitsetCache(FilterBitsetCacheConfig config) {
    Objects.requireNonNull(config, "config");
    enabled = config.enabled();
    maxBytes = config.maxBytes();
  }

  /** Whether caching is enabled. When disabled, callers build uncached, caller-owned handles. */
  synchronized boolean isEnabled() {
    return enabled;
  }

  /**
   * Informs the cache of the total byte working set of the search about to run — the sum, over the
   * segments needing a filter, of the bytes to hold one bit per vector. Called once per search
   * before any {@link #acquire}.
   *
   * <p>If the working set exceeds the fixed budget, the byte-cap eviction and oversized-entry skip
   * keep memory bounded, but repeated searches may rebuild entries. A warning is emitted once so
   * operators can raise the budget or disable the cache.
   */
  synchronized void onSearchWorkingSet(long workingSetBytes) {
    if (workingSetBytes <= 0) return;
    if (maxBytes < workingSetBytes && !workingSetWarned) {
      workingSetWarned = true;
      LOG.warning(
          "filter bitset cache maxBytes="
              + maxBytes
              + " is smaller than the current working set "
              + workingSetBytes
              + " bytes; entries will be evicted or skipped, reducing cache effectiveness");
    }
  }

  /**
   * Returns the handle for the given key, building it once if absent. The returned handle has a
   * reference taken on the caller's behalf that must be released with {@link
   * FilterBitsetHandle#decRef()} once the search completes.
   *
   * @param entryBytes byte size charged to the budget for this entry (the packed bitset length)
   */
  FilterBitsetHandle acquire(
      Query filter, Object segReaderKey, String field, long entryBytes, FilterBuilder builder)
      throws IOException {
    // A single entry larger than the whole budget is never cached; hand back a caller-owned handle.
    boolean oversized;
    synchronized (this) {
      oversized = maxBytes > 0 && entryBytes > maxBytes;
      if (oversized && !oversizedWarned) {
        oversizedWarned = true;
        LOG.warning(
            "filter bitset entry of "
                + entryBytes
                + " bytes exceeds the cache budget "
                + maxBytes
                + " bytes; serving it uncached");
      }
    }
    if (oversized) {
      return builder.build();
    }

    FilterCacheKey key = new FilterCacheKey(filter, segReaderKey, field);
    while (true) {
      CompletableFuture<FilterBitsetHandle> future;
      boolean owner = false;
      List<CompletableFuture<FilterBitsetHandle>> evicted = null;
      synchronized (this) {
        CacheEntry entry = cache.get(key);
        if (entry == null) {
          future = new CompletableFuture<>();
          cache.put(key, new CacheEntry(future, entryBytes));
          totalBytes += entryBytes;
          owner = true;
          evicted = new ArrayList<>();
          evictToBudget(key, evicted);
        } else {
          future = entry.future();
        }
      }

      if (evicted != null) {
        for (CompletableFuture<FilterBitsetHandle> f : evicted) {
          releaseCacheRef(f);
        }
      }

      if (owner) {
        // Build outside the cache lock so misses on other keys are not serialized behind this one.
        try {
          future.complete(builder.build());
        } catch (Throwable t) {
          // Drop the failed entry so a later call rebuilds, then propagate to this thread and any
          // waiters (which observe the exception via join() and retry).
          synchronized (this) {
            CacheEntry cur = cache.get(key);
            if (cur != null && cur.future() == future) {
              cache.remove(key);
              totalBytes -= cur.bytes();
            }
          }
          future.completeExceptionally(t);
          if (t instanceof IOException io) throw io;
          if (t instanceof RuntimeException re) throw re;
          if (t instanceof Error err) throw err;
          throw new RuntimeException(t);
        }
      }

      FilterBitsetHandle handle;
      try {
        handle = future.join();
      } catch (CompletionException e) {
        // The owning thread's build failed and removed the entry; retry so exactly one thread
        // rebuilds.
        continue;
      }

      if (handle.tryIncRef()) {
        return handle;
      }
      // Rare: the entry was evicted and its cache reference released between completion and our
      // acquisition. It is already gone from the map, so retry and rebuild.
    }
  }

  /**
   * Evicts eldest (least-recently-used) entries until the total size is within the byte budget and
   * the entry count is within the guard, never evicting {@code keep} (the entry needed by the
   * in-flight acquire). Runs under the cache lock; the evicted futures are collected into {@code
   * out} so their cache references are released after the lock is dropped.
   */
  private void evictToBudget(
      FilterCacheKey keep, List<CompletableFuture<FilterBitsetHandle>> out) {
    long budget = maxBytes > 0 ? maxBytes : Long.MAX_VALUE;
    var it = cache.entrySet().iterator();
    while ((totalBytes > budget || cache.size() > MAX_ENTRIES_GUARD) && it.hasNext()) {
      Map.Entry<FilterCacheKey, CacheEntry> e = it.next();
      if (e.getKey().equals(keep)) continue; // access-order puts `keep` last; never evict it
      CacheEntry ce = e.getValue();
      totalBytes -= ce.bytes();
      out.add(ce.future());
      it.remove();
    }
  }

  /**
   * Registers a one-time close listener on {@code helper} so every cache entry for this segment
   * reader is evicted (and its device allocation released) as soon as the reader's core closes —
   * e.g. when the segment is merged away or the reader is reopened with new deletes. This ties the
   * cache to the segment lifecycle (mirroring Lucene's own query cache) so obsolete entries do not
   * linger until LRU eviction.
   */
  void ensureCloseListener(IndexReader.CacheHelper helper) {
    Object segReaderKey = helper.getKey();
    synchronized (this) {
      if (!registered.add(segReaderKey)) return; // a listener is already registered for this reader
    }
    helper.addClosedListener(this::invalidateReader);
  }

  /** Evicts and releases every cache entry belonging to the given segment reader key. */
  void invalidateReader(Object segReaderKey) {
    List<CompletableFuture<FilterBitsetHandle>> toRelease = new ArrayList<>();
    synchronized (this) {
      registered.remove(segReaderKey);
      var it = cache.entrySet().iterator();
      while (it.hasNext()) {
        Map.Entry<FilterCacheKey, CacheEntry> e = it.next();
        if (Objects.equals(e.getKey().segReaderKey(), segReaderKey)) {
          CacheEntry ce = e.getValue();
          totalBytes -= ce.bytes();
          toRelease.add(ce.future());
          it.remove();
        }
      }
    }
    for (CompletableFuture<FilterBitsetHandle> future : toRelease) {
      releaseCacheRef(future);
    }
  }

  /** Evicts and releases every cache entry, resetting the byte budget accounting. */
  void clear() {
    List<CompletableFuture<FilterBitsetHandle>> toRelease = new ArrayList<>();
    synchronized (this) {
      for (CacheEntry ce : cache.values()) {
        toRelease.add(ce.future());
      }
      cache.clear();
      registered.clear();
      totalBytes = 0;
    }
    for (CompletableFuture<FilterBitsetHandle> future : toRelease) {
      releaseCacheRef(future);
    }
  }

  /** Total bytes currently charged to the budget. Test hook. */
  synchronized long currentBytesForTests() {
    return totalBytes;
  }

  /** Releases the cache's reference once the (possibly in-flight) build completes. */
  private static void releaseCacheRef(CompletableFuture<FilterBitsetHandle> future) {
    future.whenComplete(
        (handle, err) -> {
          if (handle != null) handle.close();
        });
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.carrotsearch.randomizedtesting.RandomizedRunner;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Integration tests for the reference-counting lifecycle of {@link FilterBitsetHandle} under
 * concurrent multi-partition search with real device memory. Verifies that a shared handle is safe
 * across threads, that a search after {@link FilterBitsetHandle#close()} fails cleanly, and that a
 * {@code close()} concurrent with in-flight searches never corrupts device memory.
 */
@RunWith(RandomizedRunner.class)
public class FilterBitsetHandleIT extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(FilterBitsetHandleIT.class);

  private static final int NUM_PARTITIONS = 3;
  private static final int PART_ROWS = 40;
  private static final int DIM = 16;
  private static final int NUM_QUERIES = 4;
  private static final int TOP_K = 10;
  private static final int N_ROWS = NUM_PARTITIONS * PART_ROWS;
  private static final int REMOVE_COUNT = 12; // remove global rows [0, REMOVE_COUNT)

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxAmd64());
    initializeRandom();
  }

  /** Many threads searching a shared, never-closed handle all succeed with no filtered-out rows. */
  @Test
  public void testConcurrentFilteredSearchWithSharedHandleIsStable() throws Throwable {
    final int numThreads = 8;
    final int searchesPerThread = 50;

    float[][] dataset = generateData(random, N_ROWS, DIM);
    float[][] queries = generateData(random, NUM_QUERIES, DIM);
    int[] partStart = partitionStarts();
    long[] combinedLongs = keepAfterPrefix(REMOVE_COUNT);

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      List<CagraIndex> indices = buildPartitions(dataset, partStart, resources);
      try (FilterBitsetHandle filter = FilterBitsetHandle.create(combinedLongs)) {
        // Warm up the device upload on the long-lived `resources` so the shared allocation is bound
        // to it. Per-thread resources are closed when each search thread finishes; binding the
        // allocation to one of those would free it against destroyed resources at handle close.
        assertNoFilteredRows(searchOnce(resources, indices, queries, partStart, filter), partStart);
        List<Throwable> errors =
            runConcurrentSearches(indices, queries, partStart, filter, numThreads, searchesPerThread);
        assertNoErrors(errors);
      } finally {
        closeAll(indices);
      }
    }
  }

  /** After close() releases the last reference, a subsequent search fails cleanly (not natively). */
  @Test
  public void testSearchAfterCloseThrows() throws Throwable {
    float[][] dataset = generateData(random, N_ROWS, DIM);
    float[][] queries = generateData(random, NUM_QUERIES, DIM);
    int[] partStart = partitionStarts();
    long[] combinedLongs = keepAfterPrefix(REMOVE_COUNT);

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      List<CagraIndex> indices = buildPartitions(dataset, partStart, resources);
      try {
        FilterBitsetHandle filter = FilterBitsetHandle.create(combinedLongs);
        // One search first, so the device upload has happened before the handle is released.
        searchOnce(resources, indices, queries, partStart, filter);
        filter.close();

        try {
          searchOnce(resources, indices, queries, partStart, filter);
          fail("search after close() should have thrown");
        } catch (IllegalStateException expected) {
          assertTrue(
              "unexpected message: " + expected.getMessage(),
              expected.getMessage().contains("released"));
        }
      } finally {
        closeAll(indices);
      }
    }
  }

  /**
   * A single close() concurrent with in-flight searches must not corrupt device memory: every
   * search either succeeds (with no filtered-out rows) or fails with a clean {@link
   * IllegalStateException} once the handle is fully released. Any other failure — e.g. a native CUDA
   * error from a use-after-free — fails the test.
   */
  @Test
  public void testCloseConcurrentWithSearchesDoesNotCorrupt() throws Throwable {
    final int numThreads = 8;
    final int searchesPerThread = 100;

    float[][] dataset = generateData(random, N_ROWS, DIM);
    float[][] queries = generateData(random, NUM_QUERIES, DIM);
    int[] partStart = partitionStarts();
    long[] combinedLongs = keepAfterPrefix(REMOVE_COUNT);

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      List<CagraIndex> indices = buildPartitions(dataset, partStart, resources);
      FilterBitsetHandle filter = FilterBitsetHandle.create(combinedLongs);
      // Bind the shared allocation to the long-lived `resources` (see the stable-handle test).
      searchOnce(resources, indices, queries, partStart, filter);

      List<Throwable> unexpected = new CopyOnWriteArrayList<>();
      ExecutorService pool = Executors.newFixedThreadPool(numThreads + 1);
      CountDownLatch ready = new CountDownLatch(numThreads);
      CountDownLatch go = new CountDownLatch(1);
      List<Future<?>> tasks = new ArrayList<>();
      try {
        for (int t = 0; t < numThreads; t++) {
          tasks.add(
              pool.submit(
                  () -> {
                    try (CuVSResources threadResources = CheckedCuVSResources.create()) {
                      ready.countDown();
                      go.await();
                      for (int i = 0; i < searchesPerThread; i++) {
                        try {
                          MultiPartitionSearchResults results =
                              searchOnce(threadResources, indices, queries, partStart, filter);
                          assertNoFilteredRows(results, partStart);
                        } catch (IllegalStateException released) {
                          // Clean failure after the handle was fully released — allowed.
                        }
                      }
                    } catch (Throwable other) {
                      // Any non-IllegalStateException failure (e.g. a native error from a
                      // use-after-free) is a real problem.
                      unexpected.add(other);
                    }
                    return null;
                  }));
        }
        // Closer thread: release the handle once, partway through the search storm.
        tasks.add(
            pool.submit(
                () -> {
                  ready.await();
                  go.await();
                  Thread.sleep(5);
                  filter.close();
                  return null;
                }));

        ready.await();
        go.countDown();
        for (Future<?> f : tasks) {
          try {
            f.get(60, TimeUnit.SECONDS);
          } catch (Throwable t) {
            unexpected.add(t);
          }
        }
      } finally {
        pool.shutdownNow();
        pool.awaitTermination(10, TimeUnit.SECONDS);
        filter.close(); // idempotent; ensures the allocation is released if no thread did
        closeAll(indices);
      }
      assertNoErrors(unexpected);
    }
  }

  /**
   * With fewer than {@code TOP_K} survivors, the multi-partition search cannot fill top-k, so the
   * unfilled slots (marked with the FLT_MAX sentinel distance) must be dropped rather than leaked as
   * their sentinel ordinals. Guards the distance-sentinel decode in {@code
   * MultiPartitionCagraSearchImpl}: a leaked unfilled slot would decode to a sentinel ordinal
   * (0x7FFFFFFF / 0xFFFFFFFF) and yield an out-of-range global id.
   */
  @Test
  public void testHighSelectivityDropsUnfilledSlots() throws Throwable {
    final int keepFrom = N_ROWS - 5; // keep only the last 5 rows -> fewer than TOP_K survivors

    float[][] dataset = generateData(random, N_ROWS, DIM);
    float[][] queries = generateData(random, NUM_QUERIES, DIM);
    int[] partStart = partitionStarts();
    long[] combinedLongs = keepAfterPrefix(keepFrom);

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      List<CagraIndex> indices = buildPartitions(dataset, partStart, resources);
      try (FilterBitsetHandle filter = FilterBitsetHandle.create(combinedLongs)) {
        MultiPartitionSearchResults results =
            searchOnce(resources, indices, queries, partStart, filter);

        // Every result must be a real surviving row in [keepFrom, N_ROWS); an out-of-range global
        // would mean a sentinel (unfilled) slot leaked instead of being dropped.
        for (int i = 0; i < results.count(); i++) {
          int global = partStart[results.getPartitionIndex(i)] + results.getOrdinal(i);
          assertTrue(
              "result " + global + " is not a surviving row in [" + keepFrom + ", " + N_ROWS + ")",
              global >= keepFrom && global < N_ROWS);
        }
        // Fewer than TOP_K survivors -> top-k cannot be filled, so dropping the unfilled slots
        // leaves strictly fewer than NUM_QUERIES * TOP_K results (they would total exactly that if
        // sentinels leaked)...
        assertTrue(
            "expected unfilled slots to be dropped, but got " + results.count() + " results",
            results.count() < NUM_QUERIES * TOP_K);
        // ...while still finding at least one of the survivors.
        assertTrue("expected at least one surviving row to be found", results.count() > 0);
      } finally {
        closeAll(indices);
      }
    }
  }

  // --- helpers ---

  private List<Throwable> runConcurrentSearches(
      List<CagraIndex> indices,
      float[][] queries,
      int[] partStart,
      FilterBitsetHandle filter,
      int numThreads,
      int searchesPerThread)
      throws Exception {
    List<Throwable> errors = new CopyOnWriteArrayList<>();
    ExecutorService pool = Executors.newFixedThreadPool(numThreads);
    CountDownLatch go = new CountDownLatch(1);
    List<Future<?>> tasks = new ArrayList<>();
    try {
      for (int t = 0; t < numThreads; t++) {
        tasks.add(
            pool.submit(
                () -> {
                  try (CuVSResources threadResources = CheckedCuVSResources.create()) {
                    go.await();
                    for (int i = 0; i < searchesPerThread; i++) {
                      MultiPartitionSearchResults results =
                          searchOnce(threadResources, indices, queries, partStart, filter);
                      assertNoFilteredRows(results, partStart);
                    }
                  } catch (Throwable t2) {
                    errors.add(t2);
                  }
                  return null;
                }));
      }
      go.countDown();
      for (Future<?> f : tasks) {
        f.get(60, TimeUnit.SECONDS);
      }
    } finally {
      pool.shutdownNow();
      pool.awaitTermination(10, TimeUnit.SECONDS);
    }
    return errors;
  }

  private MultiPartitionSearchResults searchOnce(
      CuVSResources resources,
      List<CagraIndex> indices,
      float[][] queries,
      int[] partStart,
      FilterBitsetHandle filter)
      throws Throwable {
    try (var queryVectors = CuVSMatrix.ofArray(queries)) {
      CagraQuery query =
          new CagraQuery.Builder(resources)
              .withTopK(TOP_K)
              .withSearchParams(new CagraSearchParams.Builder().build())
              .withQueryVectors(queryVectors)
              .build();
      return MultiPartitionCagraSearch.search(resources, indices, query, TOP_K, filter);
    }
  }

  private static void assertNoFilteredRows(MultiPartitionSearchResults results, int[] partStart) {
    for (int i = 0; i < results.count(); i++) {
      int global = partStart[results.getPartitionIndex(i)] + results.getOrdinal(i);
      assertTrue("filtered-out row appeared in results: " + global, global >= REMOVE_COUNT);
    }
  }

  private static void assertNoErrors(List<Throwable> errors) {
    if (!errors.isEmpty()) {
      for (Throwable t : errors) {
        log.error("search thread failed", t);
      }
      fail(errors.size() + " search thread(s) failed; first: " + errors.get(0));
    }
  }

  /** Combined bitset keeping global rows [prefix, N_ROWS), 64-bit word-aligned per partition. */
  private static long[] keepAfterPrefix(int prefix) {
    final int waSlice = ((PART_ROWS + 63) / 64) * 64;
    long[] combinedLongs = new long[NUM_PARTITIONS * (waSlice / 64)];
    for (int r = prefix; r < N_ROWS; r++) {
      long waBit = (long) (r / PART_ROWS) * waSlice + (r % PART_ROWS);
      combinedLongs[(int) (waBit / 64)] |= 1L << (waBit % 64);
    }
    return combinedLongs;
  }

  private static int[] partitionStarts() {
    int[] starts = new int[NUM_PARTITIONS];
    for (int p = 0; p < NUM_PARTITIONS; p++) {
      starts[p] = p * PART_ROWS;
    }
    return starts;
  }

  private List<CagraIndex> buildPartitions(
      float[][] dataset, int[] partStart, CuVSResources resources) throws Throwable {
    CagraIndexParams indexParams =
        new CagraIndexParams.Builder()
            .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
            .withGraphDegree(16)
            .withIntermediateGraphDegree(32)
            .withMetric(CuvsDistanceType.L2Expanded)
            .build();

    List<CagraIndex> indices = new ArrayList<>();
    for (int p = 0; p < partStart.length; p++) {
      float[][] slice = Arrays.copyOfRange(dataset, partStart[p], partStart[p] + PART_ROWS);
      indices.add(
          CagraIndex.newBuilder(resources).withDataset(slice).withIndexParams(indexParams).build());
    }
    return indices;
  }

  private static void closeAll(List<CagraIndex> indices) throws Exception {
    for (CagraIndex idx : indices) {
      idx.close();
    }
  }
}

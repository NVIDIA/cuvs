/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import java.util.BitSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.junit.Before;
import org.junit.Test;

/**
 * Prefilter edge cases that every index type has to handle the same way: a filter with no bits set
 * excludes every row, a filter that passes fewer rows than topK returns only those rows, and the
 * rows in a filter's last, partly used 32-bit word are honored.
 *
 * <p>The dataset has 200 rows, so its filter ends in a word that only holds rows 192 to 199. Row i
 * is the vector (i, i + 0.001, i + 0.002, ...), so the nearest row to a copy of row i is row i.
 */
public class PrefilterIT extends CuVSTestCase {

  private static final int ROWS = 200;
  private static final int DIMS = 8;

  /** Odd, so the nearest rows to a copy of row i are the ones within 5 of i, with no ties. */
  private static final int TOP_K = 11;

  private static final float[][] DATASET = rows(0, ROWS);

  /** Copies of rows 3, 100 and 197, the last of which lies in the filter's last word. */
  private static final float[][] QUERIES = {DATASET[3], DATASET[100], DATASET[197]};

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxSupportedArch());
  }

  // ---- CAGRA ------------------------------------------------------------------------------------

  /**
   * Every metric, because CAGRA marks an empty slot with FLT_MAX for L2 but with -FLT_MAX for inner
   * product, and with a different id in each search algorithm.
   */
  @Test
  public void testCagraFilterExcludingEveryRowReturnsNothing() throws Throwable {
    for (var metric : CAGRA_METRICS) {
      for (var algo : CAGRA_ALGOS) {
        List<Map<Integer, Float>> results = searchCagra(metric, algo, new BitSet(), ROWS);
        assertEquals(metric + " " + algo, List.of(Map.of(), Map.of(), Map.of()), results);
      }
    }
  }

  @Test
  public void testCagraReturnsOnlyRowsThatPassTheFilter() throws Throwable {
    BitSet filter = bits(5, 50, 192, 197, 199);
    Set<Integer> passing = filter.stream().boxed().collect(Collectors.toSet());
    for (var metric : CAGRA_METRICS) {
      for (var algo : CAGRA_ALGOS) {
        for (var result : searchCagra(metric, algo, filter, ROWS)) {
          // CAGRA is approximate, so it may not find every passing row, but it must not return
          // any other row, nor an empty slot.
          assertTrue(metric + " " + algo + ": " + result, passing.containsAll(result.keySet()));
        }
      }
    }
  }

  @Test
  public void testCagraIgnoresFilterBitsBeyondTheIndex() throws Throwable {
    BitSet filter = new BitSet();
    filter.set(ROWS, ROWS + 64);
    List<Map<Integer, Float>> results =
        searchCagra(CagraSearchParams.SearchAlgo.SINGLE_CTA, filter, ROWS + 64);
    assertEquals(List.of(Map.of(), Map.of(), Map.of()), results);
  }

  @Test
  public void testCagraRejectsFilterShorterThanTheIndex() throws Throwable {
    var e =
        assertThrows(
            IllegalArgumentException.class,
            () -> searchCagra(CagraSearchParams.SearchAlgo.SINGLE_CTA, bits(0), ROWS - 1));
    assertTrue(e.getMessage(), e.getMessage().contains("numDocs"));
  }

  // ---- Brute force ------------------------------------------------------------------------------

  @Test
  public void testBruteForceSharedFilterExcludingEveryRowReturnsNothing() throws Throwable {
    assertEquals(
        List.of(Map.of(), Map.of(), Map.of()), searchBruteForce(new BitSet[] {new BitSet()}));
  }

  @Test
  public void testBruteForcePerQueryFilters() throws Throwable {
    List<Map<Integer, Float>> results =
        searchBruteForce(new BitSet[] {new BitSet(), null, bits(197)});
    assertEquals("an empty filter excludes every row", Map.of(), results.get(0));
    assertEquals("a null filter excludes nothing", rowRange(95, 106), results.get(1).keySet());
    assertEquals(Set.of(197), results.get(2).keySet());
  }

  /**
   * Fewer passing rows than topK. Brute force picks its search path by how many rows pass: 2 of
   * 200 rows takes the sparse path, which leaves the unfilled slots unwritten, and 9 of 200 takes the
   * dense one, which fills them with rows the filter removed. Both must return only the passing
   * rows.
   */
  @Test
  public void testBruteForceReturnsOnlyRowsThatPassTheFilter() throws Throwable {
    for (BitSet filter : new BitSet[] {bits(3, 197), bits(1, 3, 7, 50, 100, 150, 192, 197, 199)}) {
      for (var result : searchBruteForce(new BitSet[] {filter})) {
        assertEquals(filter.stream().boxed().collect(Collectors.toSet()), result.keySet());
      }
    }
  }

  @Test
  public void testBruteForceTopKLargerThanTheIndex() throws Throwable {
    float[][] dataset = rows(0, 3);
    try (CuVSResources resources = CheckedCuVSResources.create();
        BruteForceIndex index =
            BruteForceIndex.newBuilder(resources)
                .withDataset(dataset)
                .withIndexParams(new BruteForceIndexParams.Builder().build())
                .build()) {
      BruteForceQuery query =
          new BruteForceQuery.Builder(resources)
              .withTopK(5)
              .withQueryVectors(new float[][] {dataset[0]})
              .build();
      assertEquals(Set.of(0, 1, 2), index.search(query).getResults().get(0).keySet());
    }
  }

  @Test
  public void testBruteForceIgnoresFiltersBeyondTheQueries() throws Throwable {
    List<Map<Integer, Float>> results =
        searchBruteForce(new BitSet[] {new BitSet(), null, bits(197), bits(5)});
    assertEquals(Map.of(), results.get(0));
    assertEquals(rowRange(95, 106), results.get(1).keySet());
    assertEquals(Set.of(197), results.get(2).keySet());
  }

  @Test
  public void testBruteForceRejectsWrongNumberOfFilters() throws Throwable {
    assertThrows(
        IllegalArgumentException.class, () -> searchBruteForce(new BitSet[] {bits(1), bits(2)}));
  }

  // ---- Tiered -----------------------------------------------------------------------------------

  @Test
  public void testTieredFilterExcludingEveryRowReturnsNothing() throws Throwable {
    for (var part : TieredPart.values()) {
      assertEquals(
          part.toString(), List.of(Map.of(), Map.of(), Map.of()), searchTiered(part, new BitSet()));
    }
  }

  @Test
  public void testTieredReturnsOnlyRowsThatPassTheFilter() throws Throwable {
    BitSet filter = bits(5, 50, 192, 197, 199);
    Set<Integer> passing = filter.stream().boxed().collect(Collectors.toSet());
    for (var result : searchTiered(TieredPart.CAGRA, filter)) {
      assertTrue(result.toString(), passing.containsAll(result.keySet()));
    }
    for (var result : searchTiered(TieredPart.BRUTE_FORCE, filter)) {
      // Brute force is exact, so it finds every passing row.
      assertEquals(passing, result.keySet());
    }
  }

  // ---- Helpers ----------------------------------------------------------------------------------

  private static final CuvsDistanceType[] CAGRA_METRICS = {
    CuvsDistanceType.L2Expanded, CuvsDistanceType.InnerProduct, CuvsDistanceType.CosineExpanded
  };

  private static final CagraSearchParams.SearchAlgo[] CAGRA_ALGOS = {
    CagraSearchParams.SearchAlgo.SINGLE_CTA, CagraSearchParams.SearchAlgo.MULTI_CTA
  };

  private static float[][] rows(int from, int to) {
    float[][] rows = new float[to - from][DIMS];
    for (int i = from; i < to; i++) {
      for (int j = 0; j < DIMS; j++) {
        rows[i - from][j] = i + 0.001f * j;
      }
    }
    return rows;
  }

  private static BitSet bits(int... set) {
    BitSet bits = new BitSet();
    for (int i : set) {
      bits.set(i);
    }
    return bits;
  }

  private static Set<Integer> rowRange(int from, int to) {
    return IntStream.range(from, to).boxed().collect(Collectors.toSet());
  }

  private List<Map<Integer, Float>> searchCagra(
      CagraSearchParams.SearchAlgo algo, BitSet filter, int numDocs) throws Throwable {
    return searchCagra(CuvsDistanceType.L2Expanded, algo, filter, numDocs);
  }

  private List<Map<Integer, Float>> searchCagra(
      CuvsDistanceType metric, CagraSearchParams.SearchAlgo algo, BitSet filter, int numDocs)
      throws Throwable {
    CagraIndexParams indexParams =
        new CagraIndexParams.Builder()
            .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
            .withGraphDegree(16)
            .withIntermediateGraphDegree(32)
            .withMetric(metric)
            .build();
    try (CuVSResources resources = CheckedCuVSResources.create();
        CagraIndex index =
            CagraIndex.newBuilder(resources)
                .withDataset(DATASET)
                .withIndexParams(indexParams)
                .build();
        var hostVectors = CuVSMatrix.ofArray(DATASET);
        var deviceVectors = hostVectors.toDevice(resources);
        // Rows of 8 floats need no padding, and an already padded matrix can only be viewed.
        var indexDataset = index.makePaddedDatasetView(deviceVectors);
        var queryVectors = CuVSMatrix.ofArray(QUERIES)) {
      index.updateDataset(indexDataset);
      CagraQuery query =
          new CagraQuery.Builder(resources)
              .withTopK(TOP_K)
              .withSearchParams(new CagraSearchParams.Builder().withAlgo(algo).build())
              .withQueryVectors(queryVectors)
              .withPrefilter(filter, numDocs)
              .build();
      return index.search(query).getResults();
    }
  }

  private List<Map<Integer, Float>> searchBruteForce(BitSet[] filters) throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create();
        BruteForceIndex index =
            BruteForceIndex.newBuilder(resources)
                .withDataset(DATASET)
                .withIndexParams(new BruteForceIndexParams.Builder().build())
                .build()) {
      BruteForceQuery query =
          new BruteForceQuery.Builder(resources)
              .withTopK(TOP_K)
              .withQueryVectors(QUERIES)
              .withPrefilters(filters, ROWS)
              .build();
      return index.search(query).getResults();
    }
  }

  /**
   * Where a tiered index keeps its rows. Its search filters the rows it has added since the CAGRA
   * build with global row numbers, so an index with rows in both parts is left out here.
   */
  private enum TieredPart {
    CAGRA,
    BRUTE_FORCE
  }

  private List<Map<Integer, Float>> searchTiered(TieredPart part, BitSet filter) throws Throwable {
    CagraIndexParams cagraParams =
        new CagraIndexParams.Builder().withGraphDegree(16).withIntermediateGraphDegree(32).build();
    TieredIndexParams indexParams =
        new TieredIndexParams.Builder()
            // A tiered index only builds a CAGRA index over more than minAnnRows rows.
            .minAnnRows(part == TieredPart.CAGRA ? 2 : 2 * ROWS)
            .withCagraParams(cagraParams)
            .build();
    try (CuVSResources resources = CheckedCuVSResources.create();
        TieredIndex index =
            TieredIndex.newBuilder(resources)
                .withDataset(DATASET)
                .withIndexParams(indexParams)
                .build()) {
      TieredIndexQuery query =
          TieredIndexQuery.newBuilder(resources)
              .withTopK(TOP_K)
              .withQueryVectors(QUERIES)
              .withSearchParams(new CagraSearchParams.Builder().build())
              .withPrefilter(filter, ROWS)
              .build();
      return index.search(query).getResults();
    }
  }
}

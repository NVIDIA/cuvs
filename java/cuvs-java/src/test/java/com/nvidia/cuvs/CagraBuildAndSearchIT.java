/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;
import static com.carrotsearch.randomizedtesting.RandomizedTest.randomIntBetween;
import static com.nvidia.cuvs.CuVSMatrixIT.assertSame2dArray;
import static org.junit.Assert.*;

import com.carrotsearch.randomizedtesting.RandomizedRunner;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import com.nvidia.cuvs.spi.CuVSProvider;
import java.lang.foreign.Arena;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.function.Function;
import java.util.function.LongToIntFunction;
import java.util.function.Supplier;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@RunWith(RandomizedRunner.class)
public class CagraBuildAndSearchIT extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(CagraBuildAndSearchIT.class);

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxAmd64());
    initializeRandom();
    log.trace("Random context initialized for test.");
  }

  private static void runConcurrently(
      boolean usePooledMemory, int nThreads, Function<Integer, Runnable> runnableSupplier)
      throws ExecutionException, InterruptedException, TimeoutException {
    try (ExecutorService parallelExecutor = Executors.newFixedThreadPool(nThreads)) {
      if (usePooledMemory) {
        CuVSProvider.provider().enableRMMPooledMemory(10, 60);
      }
      var futures = new CompletableFuture[nThreads];
      for (int j = 0; j < nThreads; j++) {
        futures[j] = CompletableFuture.runAsync(runnableSupplier.apply(j), parallelExecutor);
      }

      CompletableFuture.allOf(futures)
          .exceptionally(
              t -> {
                log.error("Exception while executing runnable", t);
                fail("Exception while executing runnable: " + unwrap(t));
                return null;
              })
          .get(2000, TimeUnit.SECONDS);

      parallelExecutor.shutdown();
      assertTrue(
          "Timeout waiting for parallelExecutor to finish",
          parallelExecutor.awaitTermination(10, TimeUnit.SECONDS));
    } finally {
      if (usePooledMemory) {
        CuVSProvider.provider().resetRMMPooledMemory();
      }
    }
  }

  private static Throwable unwrap(Throwable t) {
    var root = t;
    while (root.getCause() != null) {
      root = root.getCause();
    }
    return root;
  }

  private static void runInAnotherThread(Runnable runnable)
      throws ExecutionException, InterruptedException, TimeoutException {
    try (ExecutorService singleExecutor = Executors.newSingleThreadExecutor()) {
      singleExecutor.submit(runnable).get(2000, TimeUnit.SECONDS);
    }
  }

  private static List<Map<Integer, Float>> getExpectedResults() {
    return Arrays.asList(
        Map.of(3, 0.038782578f, 2, 0.3590463f, 0, 0.83774555f),
        Map.of(0, 0.12472608f, 2, 0.21700792f, 1, 0.31918612f),
        Map.of(3, 0.047766715f, 2, 0.20332818f, 0, 0.48305473f),
        Map.of(1, 0.15224178f, 0, 0.59063464f, 3, 0.5986642f));
  }

  private static float[][] createSampleQueries() {
    return new float[][] {
      {0.48216683f, 0.0428398f},
      {0.5084142f, 0.6545497f},
      {0.51260436f, 0.2643005f},
      {0.05198065f, 0.5789965f}
    };
  }

  private static float[][] createSampleData() {
    return new float[][] {
      {0.74021935f, 0.9209938f},
      {0.03902049f, 0.9689629f},
      {0.92514056f, 0.4463501f},
      {0.6673192f, 0.10993068f}
    };
  }

  /**
   * A basic test that checks the whole flow - from indexing to search.
   */
  @Test
  public void testIndexingAndSearchingFlow() throws Throwable {
    float[][] dataset = createSampleData();
    float[][] queries = createSampleQueries();
    List<Map<Integer, Float>> expectedResults = getExpectedResults();

    int numTestsRuns = 5;
    try (CuVSResources resources = CheckedCuVSResources.create();
        var hostVectors = CuVSMatrix.ofArray(dataset);
        var deviceVectors = hostVectors.toDevice(resources)) {
      for (int j = 0; j < numTestsRuns; j++) {
        try (var index = indexOnce(CuVSMatrix.ofArray(dataset), resources)) {
          // Serialize the host index first, so the file keeps its host layout.
          var indexPath = serializeOnce(index);
          // A host build and a graph-only deserialization both leave the index without the
          // device-padded vectors CAGRA search needs, so give each index its own padded copy.
          try (var indexDataset = index.makePaddedDataset(deviceVectors);
              var loadedIndex = deserializeOnce(indexPath, resources);
              var loadedDataset = loadedIndex.makePaddedDataset(deviceVectors)) {
            index.updateDataset(indexDataset);
            loadedIndex.updateDataset(loadedDataset);
            queryAndCompare(
                index,
                loadedIndex,
                SearchResults.IDENTITY_MAPPING,
                queries,
                expectedResults,
                resources);
            Files.deleteIfExists(indexPath);
          }
        }
      }
    }
  }

  @Test
  public void testDeserializeReturnsCallerOwnedStandardDataset() throws Throwable {
    float[][] dataset = createSampleData();
    float[][] queries = createSampleQueries();
    List<Map<Integer, Float>> expectedResults = getExpectedResults();

    try (CuVSResources resources = CheckedCuVSResources.create();
        var hostVectors = CuVSMatrix.ofArray(dataset);
        var deviceVectors = hostVectors.toDevice(resources);
        var index = indexOnce(CuVSMatrix.ofArray(dataset), resources)) {
      // Serialize the host index first, so the file keeps the host standard layout this test
      // deserializes into.
      var indexPath = serializeOnce(index);
      try (var indexDataset = index.makePaddedDataset(deviceVectors);
          var outDataset = new CagraIndex.StandardDataset();
          var inputStream = Files.newInputStream(indexPath);
          var loadedIndex = CagraIndex.newBuilder(resources).from(inputStream, outDataset).build();
          // The deserialized standard dataset is caller-owned but not searchable as-is.
          var loadedDataset = loadedIndex.makePaddedDataset(deviceVectors)) {
        assertTrue(outDataset.isPresent());
        index.updateDataset(indexDataset);
        loadedIndex.updateDataset(loadedDataset);
        queryAndCompare(
            index,
            loadedIndex,
            SearchResults.IDENTITY_MAPPING,
            queries,
            expectedResults,
            resources);
      } finally {
        Files.deleteIfExists(indexPath);
      }
    }
  }

  /**
   * A basic test that checks the whole flow - from indexing to search.
   */
  @Test
  public void testIndexingAndSearchingFlowInDifferentThreads() throws Throwable {
    float[][] dataset = createSampleData();
    float[][] queries = createSampleQueries();
    List<Map<Integer, Float>> expectedResults = getExpectedResults();

    int numTestsRuns = 5;
    try (CuVSResources resources = CheckedCuVSResources.create()) {
      for (int j = 0; j < numTestsRuns; j++) {
        runInAnotherThread(
            () -> {
              try (var index = indexOnce(CuVSMatrix.ofArray(dataset), resources);
                  var hostVectors = CuVSMatrix.ofArray(dataset);
                  var deviceVectors = hostVectors.toDevice(resources)) {
                var indexPath = serializeOnce(index);
                // Neither the host build nor the graph-only deserialization leaves device-padded
                // vectors behind, so give each index its own padded copy.
                try (var indexDataset = index.makePaddedDataset(deviceVectors);
                    var loadedIndex = deserializeOnce(indexPath, resources);
                    var loadedDataset = loadedIndex.makePaddedDataset(deviceVectors)) {
                  index.updateDataset(indexDataset);
                  loadedIndex.updateDataset(loadedDataset);
                  queryAndCompare(
                      index,
                      loadedIndex,
                      SearchResults.IDENTITY_MAPPING,
                      queries,
                      expectedResults,
                      resources);
                } finally {
                  Files.deleteIfExists(indexPath);
                }
              } catch (Throwable e) {
                throw new RuntimeException(e);
              }
            });
      }
    }
  }

  /**
   * A basic test that checks the whole flow - from indexing to search.
   */
  @Test
  public void testIndexingAndSearchingFlowConcurrently() throws Throwable {
    testIndexingAndSearchingFlowConcurrently(false);
  }

  @Test
  public void testIndexingAndSearchingFlowConcurrentlyWithPooledMemory() throws Throwable {
    testIndexingAndSearchingFlowConcurrently(true);
  }

  private void testIndexingAndSearchingFlowConcurrently(boolean usePooledMemory) throws Throwable {
    final float[][] dataset = createSampleData();
    float[][] queries = createSampleQueries();
    List<Map<Integer, Float>> expectedResults = getExpectedResults();

    int numTestsRuns = 10;

    runConcurrently(
        usePooledMemory,
        numTestsRuns,
        threadIdx ->
            () -> {
              log.debug("Indexing threadIdx:{}-{}", threadIdx, Thread.currentThread().getName());
              try (CuVSResources resources = CheckedCuVSResources.create();
                  var matrix = CuVSMatrix.ofArray(dataset);
                  var index = indexOnce(matrix, resources);
                  var hostVectors = CuVSMatrix.ofArray(dataset);
                  var deviceVectors = hostVectors.toDevice(resources)) {
                var indexPath = serializeOnce(index);
                // Neither the host build nor the graph-only deserialization leaves device-padded
                // vectors behind, so give each index its own padded copy.
                try (var indexDataset = index.makePaddedDataset(deviceVectors);
                    var loadedIndex = deserializeOnce(indexPath, resources);
                    var loadedDataset = loadedIndex.makePaddedDataset(deviceVectors)) {
                  index.updateDataset(indexDataset);
                  loadedIndex.updateDataset(loadedDataset);
                  log.debug(
                      "Querying threadIdx:{}-{}", threadIdx, Thread.currentThread().getName());
                  queryAndCompare(
                      index,
                      loadedIndex,
                      SearchResults.IDENTITY_MAPPING,
                      queries,
                      expectedResults,
                      resources);
                } finally {
                  Files.deleteIfExists(indexPath);
                }
              } catch (Throwable e) {
                throw new RuntimeException(e);
              }
              log.debug("Done threadIdx:{}-{}", threadIdx, Thread.currentThread().getName());
            });
  }

  @Test
  public void testFloatIndexing() throws Throwable {
    testIndexing(
        false,
        () ->
            CuVSMatrix.ofArray(
                createFloatMatrix(randomIntBetween(2, 1024), randomIntBetween(2, 2048))));
  }

  @Test
  public void testByteIndexing() throws Throwable {
    testIndexing(
        false,
        () ->
            CuVSMatrix.ofArray(
                createByteMatrix(randomIntBetween(2, 1024), randomIntBetween(2, 2048))));
  }

  @Test
  public void testFloatIndexingWithPooledMemory() throws Throwable {
    testIndexing(
        true,
        () ->
            CuVSMatrix.ofArray(
                createFloatMatrix(randomIntBetween(2, 1024), randomIntBetween(2, 2048))));
  }

  @Test
  public void testByteIndexingWithPooledMemory() throws Throwable {
    testIndexing(
        true,
        () ->
            CuVSMatrix.ofArray(
                createByteMatrix(randomIntBetween(2, 1024), randomIntBetween(2, 2048))));
  }

  private void testIndexing(boolean usePooledMemory, Supplier<CuVSMatrix> matrixFactory)
      throws Exception {
    for (int i = 0; i < 10; ++i) {
      try (var dataset = matrixFactory.get()) {
        int numRunners = 4;
        final int iteration = i;
        runConcurrently(
            usePooledMemory,
            numRunners,
            threadIdx ->
                () -> {
                  try (CuVSResources resources = CheckedCuVSResources.create()) {
                    // Create a local reference to the dataset, as index will close the dataset too
                    // when it gets closed.
                    var indexDatasetReference = dataset.toHost();
                    log.debug(
                        "Indexing iteration:{} threadIdx:{} dataset:{}",
                        iteration,
                        threadIdx,
                        dataset);
                    var index = indexOnce(indexDatasetReference, resources);
                    log.debug("Done {} {}", iteration, threadIdx);
                    index.close();
                  } catch (Throwable e) {
                    throw new RuntimeException(e);
                  }
                });
      }
    }
  }

  @Test
  public void testFloatSerialization() throws Throwable {
    testSerialization(
        false,
        () ->
            CuVSMatrix.ofArray(
                createFloatMatrix(randomIntBetween(2, 1024), randomIntBetween(2, 2048))));
  }

  @Test
  public void testByteSerialization() throws Throwable {
    testSerialization(
        false,
        () ->
            CuVSMatrix.ofArray(
                createByteMatrix(randomIntBetween(2, 1024), randomIntBetween(2, 2048))));
  }

  @Test
  public void testFloatSerializationWithPooledMemory() throws Throwable {
    testSerialization(
        true,
        () ->
            CuVSMatrix.ofArray(
                createFloatMatrix(randomIntBetween(2, 1024), randomIntBetween(2, 2048))));
  }

  @Test
  public void testByteSerializationWithPooledMemory() throws Throwable {
    testSerialization(
        true,
        () ->
            CuVSMatrix.ofArray(
                createByteMatrix(randomIntBetween(2, 1024), randomIntBetween(2, 2048))));
  }

  private void testSerialization(boolean usePooledMemory, Supplier<CuVSMatrix> matrixFactory)
      throws Throwable {
    for (int i = 0; i < 10; ++i) {
      try (final var dataset = matrixFactory.get()) {
        int numRunners = 4;
        runConcurrently(
            usePooledMemory,
            numRunners,
            threadIdx ->
                () -> {
                  // Create a local reference to the dataset, as index will close the dataset too
                  // when it gets closed.
                  var indexDatasetReference = dataset.toHost();
                  try (CuVSResources resources = CheckedCuVSResources.create();
                      var index = indexOnce(indexDatasetReference, resources)) {
                    var indexPath = serializeOnce(index);
                    Files.deleteIfExists(indexPath);
                  } catch (Throwable e) {
                    throw new RuntimeException(e);
                  }
                });
      }
    }
  }

  @Test
  public void testFloatDeserialization() throws Throwable {
    testDeserialization(
        false,
        () ->
            CuVSMatrix.ofArray(
                createFloatMatrix(randomIntBetween(2, 1024), randomIntBetween(2, 2048))));
  }

  @Test
  public void testByteDeserialization() throws Throwable {
    testDeserialization(
        false,
        () ->
            CuVSMatrix.ofArray(
                createByteMatrix(randomIntBetween(2, 1024), randomIntBetween(2, 2048))));
  }

  @Test
  public void testFloatDeserializationWithPooledMemory() throws Throwable {
    testDeserialization(
        true,
        () ->
            CuVSMatrix.ofArray(
                createFloatMatrix(randomIntBetween(2, 1024), randomIntBetween(2, 2048))));
  }

  @Test
  public void testByteDeserializationWithPooledMemory() throws Throwable {
    testDeserialization(
        true,
        () ->
            CuVSMatrix.ofArray(
                createByteMatrix(randomIntBetween(2, 1024), randomIntBetween(2, 2048))));
  }

  private void testDeserialization(boolean usePooledMemory, Supplier<CuVSMatrix> matrixFactory)
      throws Throwable {
    Path indexPath;
    try (var dataset = matrixFactory.get()) {
      indexPath = createSerializedIndex(dataset);
    }
    try {
      for (int i = 0; i < 10; ++i) {
        int numTestsRuns = 4;
        runConcurrently(
            usePooledMemory,
            numTestsRuns,
            threadIdx ->
                () -> {
                  try (CuVSResources resources = CheckedCuVSResources.create();
                      var loadedIndex = deserializeOnce(indexPath, resources)) {
                    // just validate deserialize/close path under concurrency
                  } catch (Throwable e) {
                    throw new RuntimeException(e);
                  }
                });
      }
    } finally {
      Files.deleteIfExists(indexPath);
    }
  }

  private Path createSerializedIndex(CuVSMatrix dataset) throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create();
        var index = indexOnce(dataset, resources)) {
      return serializeOnce(index);
    }
  }

  @Test
  public void testReconstructIndexFromGraph() throws Throwable {
    float[][] sampleData = createSampleData();
    try (var dataset = CuVSMatrix.ofArray(sampleData)) {
      var queries = createSampleQueries();
      List<Map<Integer, Float>> expectedResults = getExpectedResults();

      try (CuVSResources resources = CuVSResources.create();
          var hostVectors = CuVSMatrix.ofArray(sampleData);
          // Reconstructing from a graph needs the vectors on device.
          var deviceVectors = hostVectors.toDevice(resources);
          var index = indexOnce(dataset, resources);
          // getGraph() returns a view of the index's own graph, and a device graph is kept as a
          // view rather than copied. Hand over a host copy so the reconstruction owns its graph
          // and survives updates to the index it came from.
          var graph = index.getGraph().toHost()) {

        try (var reconstructedIndex =
                CagraIndex.newBuilder(resources)
                    .from(graph)
                    .withDataset(deviceVectors)
                    .withIndexParams(
                        new CagraIndexParams.Builder()
                            .withMetric(CuvsDistanceType.L2Expanded)
                            .build())
                    .build();
            var indexDataset = index.makePaddedDataset(deviceVectors);
            var reconstructedDataset = reconstructedIndex.makePaddedDataset(deviceVectors)) {
          index.updateDataset(indexDataset);
          reconstructedIndex.updateDataset(reconstructedDataset);

          queryAndCompare(
              index,
              reconstructedIndex,
              SearchResults.IDENTITY_MAPPING,
              queries,
              expectedResults,
              resources);

          var originalIndexPath = serializeOnce(index);
          var reconstructedIndexPath = serializeOnce(reconstructedIndex);

          var originalBytes = Files.readAllBytes(originalIndexPath);
          var reconstructedBytes = Files.readAllBytes(reconstructedIndexPath);

          assertArrayEquals(originalBytes, reconstructedBytes);

          Files.deleteIfExists(originalIndexPath);
          Files.deleteIfExists(reconstructedIndexPath);
        }
      }
    }
  }

  @Test
  public void testIndexingAndSearchingFlowWithCustomMappingFunction() throws Throwable {
    float[][] sampleData = createSampleData();
    var dataset = CuVSMatrix.ofArray(sampleData);
    float[][] queries = createSampleQueries();
    var expectedResults =
        List.of(
            Map.of(0, 0.038782578f, 3, 0.3590463f, 1, 0.83774555f),
            Map.of(1, 0.12472608f, 3, 0.21700792f, 2, 0.31918612f),
            Map.of(0, 0.047766715f, 3, 0.20332818f, 1, 0.48305473f),
            Map.of(2, 0.15224178f, 1, 0.59063464f, 0, 0.5986642f));

    LongToIntFunction rotate = l -> (int) ((l + 1) % dataset.size());
    try (CuVSResources resources = CheckedCuVSResources.create();
        var index = indexOnce(dataset, resources);
        var hostVectors = CuVSMatrix.ofArray(sampleData);
        var deviceVectors = hostVectors.toDevice(resources)) {
      var indexPath = serializeOnce(index);
      try (var indexDataset = index.makePaddedDataset(deviceVectors);
          var loadedIndex = deserializeOnce(indexPath, resources);
          var loadedDataset = loadedIndex.makePaddedDataset(deviceVectors)) {
        index.updateDataset(indexDataset);
        loadedIndex.updateDataset(loadedDataset);
        queryAndCompare(index, loadedIndex, rotate, queries, expectedResults, resources);
      } finally {
        Files.deleteIfExists(indexPath);
      }
    }
  }

  @Test
  public void testIndexingAndSearchingFlowWithCustomMappingList() throws Throwable {
    float[][] sampleData = createSampleData();
    var dataset = CuVSMatrix.ofArray(sampleData);
    float[][] queries = createSampleQueries();
    var mappings = List.of(4, 3, 2, 1);
    var expectedResults =
        List.of(
            Map.of(1, 0.038782578f, 2, 0.3590463f, 4, 0.83774555f),
            Map.of(4, 0.12472608f, 2, 0.21700792f, 3, 0.31918612f),
            Map.of(1, 0.047766715f, 2, 0.20332818f, 4, 0.48305473f),
            Map.of(3, 0.15224178f, 4, 0.59063464f, 1, 0.5986642f));

    LongToIntFunction rotate = SearchResults.mappingsFromList(mappings);
    try (CuVSResources resources = CheckedCuVSResources.create();
        var index = indexOnce(dataset, resources);
        var hostVectors = CuVSMatrix.ofArray(sampleData);
        var deviceVectors = hostVectors.toDevice(resources)) {
      var indexPath = serializeOnce(index);
      try (var indexDataset = index.makePaddedDataset(deviceVectors);
          var loadedIndex = deserializeOnce(indexPath, resources);
          var loadedDataset = loadedIndex.makePaddedDataset(deviceVectors)) {
        index.updateDataset(indexDataset);
        loadedIndex.updateDataset(loadedDataset);
        queryAndCompare(index, loadedIndex, rotate, queries, expectedResults, resources);
      } finally {
        Files.deleteIfExists(indexPath);
      }
    }
  }

  /**
   * A test that checks the pre-filtering feature.
   */
  @Test
  public void testPrefilteringReducesResults() throws Throwable {

    // Sample data and query
    float[][] dataset = createSampleData();
    float[][] queries = {{0.48216683f, 0.0428398f}};

    // Expected search results
    List<Map<Integer, Float>> expectedResults = List.of(Map.of(3, 0.038782578f, 2, 0.3590463f));

    // Expected filtered search results
    List<Map<Integer, Float>> expectedFilteredResults =
        List.of(Map.of(2, 0.3590463f, 0, 0.83774555f));

    CagraIndexParams indexParams =
        new CagraIndexParams.Builder()
            .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
            .withGraphDegree(2)
            .withIntermediateGraphDegree(4)
            .withNumWriterThreads(2)
            .withMetric(CuvsDistanceType.L2Expanded)
            .build();

    try (CuVSResources resources = CheckedCuVSResources.create();
        CagraIndex index =
            CagraIndex.newBuilder(resources)
                .withDataset(dataset)
                .withIndexParams(indexParams)
                .build();
        var hostVectors = CuVSMatrix.ofArray(dataset);
        var deviceVectors = hostVectors.toDevice(resources);
        var indexDataset = index.makePaddedDataset(deviceVectors)) {

      index.updateDataset(indexDataset);

      // No prefilter (all points allowed)
      // Pin SINGLE_CTA; AUTO may pick MULTI_CTA, which drops neighbors on this tiny dataset.
      CagraSearchParams searchParams =
          new CagraSearchParams.Builder().withAlgo(CagraSearchParams.SearchAlgo.SINGLE_CTA).build();

      // No prefilter (all points allowed)
      try (var queryVectors = CuVSMatrix.ofArray(queries)) {
        CagraQuery fullQuery =
            new CagraQuery.Builder(resources)
                .withTopK(2)
                .withSearchParams(searchParams)
                .withQueryVectors(queryVectors)
                .build();

        SearchResults fullSearchResults = index.search(fullQuery);
        List<Map<Integer, Float>> fullResults = fullSearchResults.getResults();
        log.debug("Full results: {}", fullResults);

        // Apply prefilter: only allow ids 0 and 2 (bitset: 1100)
        BitSet prefilter = new BitSet(4);
        prefilter.set(0);
        prefilter.set(2);

        CagraQuery filteredQuery =
            new CagraQuery.Builder(resources)
                .withTopK(2)
                .withSearchParams(searchParams)
                .withQueryVectors(queryVectors)
                .withPrefilter(prefilter, 4)
                .build();

        SearchResults filteredSearchResults = index.search(filteredQuery);
        List<Map<Integer, Float>> filteredResults = filteredSearchResults.getResults();
        log.debug("Filtered results: {}", filteredResults);

        assertEquals(expectedResults, fullResults);
        assertEquals(expectedFilteredResults, filteredResults);
      }
    }
  }

  /**
   * Regression test for a bug in {@code CagraIndexImpl.search}: the {@code distances} output tensor
   * was described (dtype and buffer size) using the <em>query</em> vectors' data type instead of the
   * float32 type the C API mandates. Float queries masked the bug because their dtype already matches
   * float32. With byte (uint8) queries the distances tensor became uint8/8-bit and its device buffer
   * was sized at 1 byte per value instead of 4, which the C wrapper rejects up front
   * ("distances should be of type float32") -- and would otherwise be a GPU buffer overflow. The
   * existing byte tests only cover build/serialize/deserialize, never search, so this path was
   * uncovered.
   */
  @Test
  public void testByteQuerySearch() throws Throwable {
    // Small, unambiguous byte dataset (values within signed-byte range, mapped to uint8).
    byte[][] dataset = {
      {0, 0},
      {5, 5},
      {50, 50},
      {100, 100}
    };
    // Each query equals a dataset row, so its nearest neighbor is that row at distance 0.
    byte[][] queries = {
      {0, 0}, // -> id 0
      {100, 100} // -> id 3
    };
    List<Map<Integer, Float>> expectedResults = List.of(Map.of(0, 0.0f), Map.of(3, 0.0f));

    try (CuVSResources resources = CheckedCuVSResources.create();
        var index = indexOnce(CuVSMatrix.ofArray(dataset), resources);
        var hostVectors = CuVSMatrix.ofArray(dataset);
        var deviceVectors = hostVectors.toDevice(resources);
        var indexDataset = index.makePaddedDataset(deviceVectors);
        var queryVectors = CuVSMatrix.ofArray(queries)) {
      index.updateDataset(indexDataset);

      CagraQuery query =
          new CagraQuery.Builder(resources)
              .withTopK(1)
              // Pin SINGLE_CTA; AUTO may pick MULTI_CTA, which drops neighbors on this tiny
              // dataset.
              .withSearchParams(
                  new CagraSearchParams.Builder()
                      .withAlgo(CagraSearchParams.SearchAlgo.SINGLE_CTA)
                      .build())
              .withQueryVectors(queryVectors)
              .withMapping(SearchResults.IDENTITY_MAPPING)
              .build();

      // Fails with "distances should be of type float32" when the bug is present.
      SearchResults results = index.search(query);
      log.debug("Byte-query search results: {}", results.getResults());
      checkResults(expectedResults, results.getResults());
    }
  }

  private CagraIndex indexOnce(CuVSMatrix dataset, CuVSResources resources) throws Throwable {
    // Configure index parameters
    CagraIndexParams indexParams =
        new CagraIndexParams.Builder()
            .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
            .withGraphDegree(1)
            .withIntermediateGraphDegree(2)
            .withNumWriterThreads(32)
            .withMetric(CuvsDistanceType.L2Expanded)
            .build();

    // Create the index with the dataset
    return CagraIndex.newBuilder(resources)
        .withDataset(dataset)
        .withIndexParams(indexParams)
        .build();
  }

  private Path serializeOnce(CagraIndex index) throws Throwable {
    // Saving the index on to the disk.
    var indexFilePath = Path.of(UUID.randomUUID() + ".cag");
    try (var outputStream = Files.newOutputStream(indexFilePath)) {
      index.serialize(outputStream);
    }
    return indexFilePath;
  }

  private CagraIndex deserializeOnce(Path indexFilePath, CuVSResources resources) throws Throwable {
    // Loading a CAGRA index from disk.
    try (var inputStream = Files.newInputStream(indexFilePath)) {
      return CagraIndex.newBuilder(resources).from(inputStream).build();
    }
  }

  private void queryAndCompare(
      CagraIndex index1,
      CagraIndex index2,
      LongToIntFunction mapping,
      float[][] queries,
      List<Map<Integer, Float>> expectedResults,
      CuVSResources resources)
      throws Throwable {
    // Configure search parameters.
    // Pin SINGLE_CTA; AUTO may pick MULTI_CTA, which drops neighbors on this tiny dataset.
    CagraSearchParams searchParams =
        new CagraSearchParams.Builder().withAlgo(CagraSearchParams.SearchAlgo.SINGLE_CTA).build();

    // Create a query object with the query vectors
    try (var queryVectors = CuVSMatrix.ofArray(queries)) {
      CagraQuery cuvsQuery =
          new CagraQuery.Builder(resources)
              .withTopK(3)
              .withSearchParams(searchParams)
              .withQueryVectors(queryVectors)
              .withMapping(mapping)
              .build();

      // Perform the search
      SearchResults results = index1.search(cuvsQuery);

      // Check results
      log.debug(results.getResults().toString());
      checkResults(expectedResults, results.getResults());

      // Search from the second index
      results = index2.search(cuvsQuery);

      // Check results
      log.debug(results.getResults().toString());
      checkResults(expectedResults, results.getResults());
    }
  }

  /**
   * Tests that an index built starting from a native MemorySegment is identical to one built from
   * Java heap arrays
   */
  @Test
  public void testNativeDatasetEquivalent() throws Throwable {
    float[][] sampleData = createSampleData();
    float[][] queries = createSampleQueries();
    List<Map<Integer, Float>> expectedResults = getExpectedResults();

    ValueLayout.OfFloat C_FLOAT =
        (ValueLayout.OfFloat) Linker.nativeLinker().canonicalLayouts().get("float");

    int rows = sampleData.length;
    int cols = sampleData[0].length;
    MemoryLayout dataMemoryLayout = MemoryLayout.sequenceLayout((long) rows * cols, C_FLOAT);

    try (Arena arena = Arena.ofShared()) {
      MemorySegment dataMemorySegment = arena.allocate(dataMemoryLayout);
      for (int r = 0; r < rows; r++) {
        MemorySegment.copy(
            sampleData[r], 0, dataMemorySegment, C_FLOAT, (r * cols * C_FLOAT.byteSize()), cols);
      }

      try (var resources = CuVSResources.create();
          var javaDataset = CuVSMatrix.ofArray(sampleData);
          var nativeDataset =
              DatasetHelper.fromMemorySegment(
                  dataMemorySegment, rows, cols, CuVSMatrix.DataType.FLOAT);
          // Indexing with an on-heap and native datasets produce the same results
          var javaIndex = indexOnce(javaDataset, resources);
          var nativeIndex = indexOnce(nativeDataset, resources);
          var deviceVectors = CuVSMatrix.ofArray(sampleData).toDevice(resources);
          var javaIndexDataset = javaIndex.makePaddedDataset(deviceVectors);
          var nativeIndexDataset = nativeIndex.makePaddedDataset(deviceVectors)) {
        javaIndex.updateDataset(javaIndexDataset);
        nativeIndex.updateDataset(nativeIndexDataset);
        queryAndCompare(
            javaIndex,
            nativeIndex,
            SearchResults.IDENTITY_MAPPING,
            queries,
            expectedResults,
            resources);
      }
    }
  }

  /**
   * Tests that an index built starting from device memory ({@link CuVSDeviceMatrix}) is identical to one
   * built from Java heap arrays
   */
  @Test
  public void testDeviceDatasetEquivalent() throws Throwable {
    float[][] sampleData = createSampleData();

    try (var resources = CuVSResources.create();
        var javaDataset = CuVSMatrix.ofArray(sampleData);
        var deviceDataset = javaDataset.toDevice(resources)) {

      // Indexing with an on-heap and native datasets produce the same results
      var javaIndex = indexOnce(javaDataset, resources);
      var deviceIndex = indexOnce(deviceDataset, resources);

      int size = (int) javaIndex.getGraph().size();
      assertEquals(size, (int) deviceIndex.getGraph().size());

      int columns = (int) javaIndex.getGraph().columns();
      assertEquals(columns, (int) deviceIndex.getGraph().columns());

      var javaIndexGraph = new int[size][columns];
      var deviceIndexGraph = new int[size][columns];
      javaIndex.getGraph().toArray(javaIndexGraph);
      deviceIndex.getGraph().toArray(deviceIndexGraph);

      assertSame2dArray(size, columns, javaIndexGraph, deviceIndexGraph);
    }
  }

  @Test
  public void testMergingIndexes() throws Throwable {
    float[][] vector1 = {
      {0.0f, 0.0f},
      {1.0f, 1.0f}
    };

    float[][] vector2 = {
      {10.0f, 10.0f},
      {11.0f, 11.0f}
    };

    float[][] queries = {
      {1.0f, 1.0f}, // Should be closest to vector1[1] -> index 1
      {10.5f, 10.5f}, // Should be closest to vector2[0] -> index 2
      {0.0f, 0.0f} // Should be closest to vector1[0] -> index 0
    };

    // Expected search results for each query (nearest neighbor and its distance)
    List<Map<Integer, Float>> expectedResults =
        Arrays.asList(
            Map.of(1, 0.0f, 0, 2.0f, 2, 162.0f),
            Map.of(2, 0.5f, 3, 0.5f, 1, 180.5f),
            Map.of(0, 0.0f, 1, 2.0f, 2, 200.0f));

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      CagraIndexParams indexParams =
          new CagraIndexParams.Builder()
              .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
              .withGraphDegree(1)
              .withIntermediateGraphDegree(2)
              .withNumWriterThreads(4)
              .withMetric(CuvsDistanceType.L2Expanded)
              .build();

      log.trace("Building first index...");
      CagraIndex index1 =
          CagraIndex.newBuilder(resources)
              .withDataset(vector1)
              .withIndexParams(indexParams)
              .build();

      log.trace("Building second index...");
      CagraIndex index2 =
          CagraIndex.newBuilder(resources)
              .withDataset(vector2)
              .withIndexParams(indexParams)
              .build();

      // Host-built indexes are not mergeable. Dim=2 is not 16-byte aligned, so upload to device,
      // allocate owning padded copies, and attach them before merge. Keep them alive until the
      // inputs are closed.
      try (var device1 = CuVSMatrix.ofArray(vector1).toDevice(resources);
          var device2 = CuVSMatrix.ofArray(vector2).toDevice(resources);
          var padded1 = index1.makePaddedDataset(device1);
          var padded2 = index2.makePaddedDataset(device2)) {
        index1.updateDataset(padded1);
        index2.updateDataset(padded2);

        log.trace("Merging indexes...");
        CagraIndex mergedIndex = CagraIndex.merge(new CagraIndex[] {index1, index2});
        log.trace("Merge completed successfully");

        // Pin SINGLE_CTA; AUTO may pick MULTI_CTA, which drops neighbors on this tiny dataset.
        CagraSearchParams searchParams =
            new CagraSearchParams.Builder()
                .withAlgo(CagraSearchParams.SearchAlgo.SINGLE_CTA)
                .build();

        try (var queryVectors = CuVSMatrix.ofArray(queries)) {
          CagraQuery query =
              new CagraQuery.Builder(resources)
                  .withTopK(3)
                  .withSearchParams(searchParams)
                  .withQueryVectors(queryVectors)
                  .withMapping(SearchResults.IDENTITY_MAPPING)
                  .build();

          log.trace("Searching merged index...");
          SearchResults results = mergedIndex.search(query);
          log.debug("Search results: " + results.getResults().toString());

          assertEquals(expectedResults, results.getResults());

          // --- Serialization/deserialization check ---
          String indexFileName = UUID.randomUUID() + ".cag";
          var indexFile = Path.of(indexFileName);

          try (var out = Files.newOutputStream(indexFile)) {
            mergedIndex.serialize(out);
          }

          try (var inputStream = Files.newInputStream(indexFile);
              CagraIndex loadedMergedIndex =
                  CagraIndex.newBuilder(resources).from(inputStream).build()) {

            SearchResults resultsFromLoaded = loadedMergedIndex.search(query);
            assertEquals(expectedResults, resultsFromLoaded.getResults());
            mergedIndex.close();
          } finally {
            Files.deleteIfExists(indexFile);
          }
          index1.close();
          index2.close();
        }
      }
    }
  }

  /**
   * Merges two indexes through a row filter and checks that the merged index holds exactly the rows
   * whose bit was set, packed together in the order the inputs were given.
   */
  @Test
  public void testFilteredMerge() throws Throwable {
    float[][] vector1 = {
      {0.0f, 0.0f},
      {1.0f, 1.0f},
      {2.0f, 2.0f}
    };

    float[][] vector2 = {
      {10.0f, 10.0f},
      {11.0f, 11.0f},
      {12.0f, 12.0f}
    };

    // Bits 0 to 2 address vector1 and bits 3 to 5 address vector2. Drop the middle row of each,
    // which leaves four rows that have to end up at positions 0 to 3 of the merged index.
    BitSet rowFilter = new BitSet();
    rowFilter.set(0, 6);
    rowFilter.clear(1);
    rowFilter.clear(4);

    float[][] survivingRows = {
      {0.0f, 0.0f},
      {2.0f, 2.0f},
      {10.0f, 10.0f},
      {12.0f, 12.0f}
    };
    // A dropped vector is no longer in the index, so its nearest neighbour is two units away.
    float[][] droppedRows = {
      {1.0f, 1.0f},
      {11.0f, 11.0f}
    };

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      CagraIndexParams indexParams =
          new CagraIndexParams.Builder()
              .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
              .withGraphDegree(1)
              .withIntermediateGraphDegree(2)
              .withNumWriterThreads(4)
              .withMetric(CuvsDistanceType.L2Expanded)
              .build();

      CagraIndex index1 =
          CagraIndex.newBuilder(resources)
              .withDataset(vector1)
              .withIndexParams(indexParams)
              .build();
      CagraIndex index2 =
          CagraIndex.newBuilder(resources)
              .withDataset(vector2)
              .withIndexParams(indexParams)
              .build();

      // Host-built indexes are not mergeable. Dim=2 is not 16-byte aligned, so upload to device,
      // allocate owning padded copies, and attach them before merge.
      try (var device1 = CuVSMatrix.ofArray(vector1).toDevice(resources);
          var device2 = CuVSMatrix.ofArray(vector2).toDevice(resources);
          var padded1 = index1.makePaddedDataset(device1);
          var padded2 = index2.makePaddedDataset(device2)) {
        index1.updateDataset(padded1);
        index2.updateDataset(padded2);

        assertEquals("Input index sizes", 3, index1.size());
        assertEquals("Input index sizes", 3, index2.size());

        try (CagraIndex mergedIndex =
            CagraIndex.merge(new CagraIndex[] {index1, index2}, null, rowFilter)) {
          assertEquals(
              "The merged index should hold one row per set bit",
              rowFilter.cardinality(),
              mergedIndex.size());

          // Pin SINGLE_CTA; AUTO may pick MULTI_CTA, which drops neighbors on this tiny dataset.
          CagraSearchParams searchParams =
              new CagraSearchParams.Builder()
                  .withAlgo(CagraSearchParams.SearchAlgo.SINGLE_CTA)
                  .build();

          try (var queryVectors = CuVSMatrix.ofArray(survivingRows)) {
            CagraQuery query =
                new CagraQuery.Builder(resources)
                    .withTopK(1)
                    .withSearchParams(searchParams)
                    .withQueryVectors(queryVectors)
                    .withMapping(SearchResults.IDENTITY_MAPPING)
                    .build();

            List<Map<Integer, Float>> results = mergedIndex.search(query).getResults();
            assertEquals(survivingRows.length, results.size());
            for (int row = 0; row < survivingRows.length; row++) {
              Map<Integer, Float> hit = results.get(row);
              assertEquals("Expected a single neighbour for row " + row, 1, hit.size());
              int id = hit.keySet().iterator().next();
              assertEquals("Surviving row " + row + " moved", row, id);
              assertEquals(
                  "Surviving row " + row + " is not an exact match", 0.0f, hit.get(id), 1e-5f);
            }
          }

          try (var queryVectors = CuVSMatrix.ofArray(droppedRows)) {
            CagraQuery query =
                new CagraQuery.Builder(resources)
                    .withTopK(1)
                    .withSearchParams(searchParams)
                    .withQueryVectors(queryVectors)
                    .withMapping(SearchResults.IDENTITY_MAPPING)
                    .build();

            List<Map<Integer, Float>> results = mergedIndex.search(query).getResults();
            assertEquals(droppedRows.length, results.size());
            for (int row = 0; row < droppedRows.length; row++) {
              Map<Integer, Float> hit = results.get(row);
              assertEquals("Expected a single neighbour for dropped row " + row, 1, hit.size());
              int id = hit.keySet().iterator().next();
              assertEquals(
                  "Dropped row " + row + " is still in the merged index", 2.0f, hit.get(id), 1e-5f);
            }
          }
        }
        index1.close();
        index2.close();
      }
    }
  }

  /**
   * A filter that selects rows beyond the ones the indexes hold is a caller mistake, not something
   * to pass on to cuVS.
   */
  @Test
  public void testFilteredMergeRejectsOversizedFilter() throws Throwable {
    float[][] vectors = {
      {0.0f, 0.0f},
      {1.0f, 1.0f}
    };

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      CagraIndexParams indexParams =
          new CagraIndexParams.Builder()
              .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
              .withGraphDegree(1)
              .withIntermediateGraphDegree(2)
              .withMetric(CuvsDistanceType.L2Expanded)
              .build();

      try (CagraIndex index =
          CagraIndex.newBuilder(resources)
              .withDataset(vectors)
              .withIndexParams(indexParams)
              .build()) {
        // The index holds two rows, so bit 2 is one row past the end of the merge.
        BitSet rowFilter = new BitSet();
        rowFilter.set(0, 3);

        assertThrows(
            IllegalArgumentException.class,
            () -> CagraIndex.merge(new CagraIndex[] {index}, null, rowFilter));
      }
    }
  }

  /**
   * A merge that fails inside cuVS has to release the index it was building, and leave every input
   * index untouched and still usable. Host-backed indexes are not mergeable, which fails the native
   * call after the output index has been allocated - the one path where the handle used to be
   * dropped on the floor. Repeating it is what would surface a release that goes too far, such as
   * one freeing an input index or the same handle twice.
   */
  @Test
  public void testFailedMergeLeavesTheInputsUsable() throws Throwable {
    float[][] vector1 = {
      {0.0f, 0.0f},
      {1.0f, 1.0f},
      {2.0f, 2.0f}
    };

    float[][] vector2 = {
      {10.0f, 10.0f},
      {11.0f, 11.0f},
      {12.0f, 12.0f}
    };

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      CagraIndexParams indexParams =
          new CagraIndexParams.Builder()
              .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
              .withGraphDegree(1)
              .withIntermediateGraphDegree(2)
              .withMetric(CuvsDistanceType.L2Expanded)
              .build();

      try (CagraIndex index1 =
              CagraIndex.newBuilder(resources)
                  .withDataset(vector1)
                  .withIndexParams(indexParams)
                  .build();
          CagraIndex index2 =
              CagraIndex.newBuilder(resources)
                  .withDataset(vector2)
                  .withIndexParams(indexParams)
                  .build()) {

        // No device dataset is attached, so cuVS refuses to merge these.
        for (int attempt = 0; attempt < 5; attempt++) {
          assertThrows(
              Throwable.class, () -> CagraIndex.merge(new CagraIndex[] {index1, index2}, null));
        }

        // The failures left the inputs alone: they still report their rows, and a merge that is
        // set up correctly still succeeds afterwards.
        assertEquals("index1 survived the failed merges", 3, index1.size());
        assertEquals("index2 survived the failed merges", 3, index2.size());

        try (var device1 = CuVSMatrix.ofArray(vector1).toDevice(resources);
            var device2 = CuVSMatrix.ofArray(vector2).toDevice(resources);
            var padded1 = index1.makePaddedDataset(device1);
            var padded2 = index2.makePaddedDataset(device2)) {
          index1.updateDataset(padded1);
          index2.updateDataset(padded2);

          try (CagraIndex mergedIndex = CagraIndex.merge(new CagraIndex[] {index1, index2}, null)) {
            assertEquals("The merged index holds every row of both inputs", 6, mergedIndex.size());

            try (var queryVectors = CuVSMatrix.ofArray(new float[][] {{0.0f, 0.0f}})) {
              CagraQuery query =
                  new CagraQuery.Builder(resources)
                      .withTopK(1)
                      .withSearchParams(
                          new CagraSearchParams.Builder()
                              .withAlgo(CagraSearchParams.SearchAlgo.SINGLE_CTA)
                              .build())
                      .withQueryVectors(queryVectors)
                      .withMapping(SearchResults.IDENTITY_MAPPING)
                      .build();

              List<Map<Integer, Float>> results = mergedIndex.search(query).getResults();
              assertEquals(1, results.size());
              assertEquals(
                  "The first row of the merge is the nearest neighbour of the first vector",
                  0,
                  (int) results.getFirst().keySet().iterator().next());
            }
          }
        }
      }
    }
  }

  // Commented out test for Logical merge strategy as it is not yet implemented in C yet
  @Test
  public void testMergeStrategies() throws Throwable {
    float[][] vector1 = {
      {0.0f, 0.0f},
      {1.0f, 1.0f}
    };

    float[][] vector2 = {
      {10.0f, 10.0f},
      {11.0f, 11.0f}
    };

    float[][] queries = {
      {1.0f, 1.0f},
      {10.5f, 10.5f},
      {0.0f, 0.0f}
    };

    List<Map<Integer, Float>> expectedResults =
        Arrays.asList(
            Map.of(1, 0.0f, 0, 2.0f, 2, 162.0f),
            Map.of(2, 0.5f, 3, 0.5f, 1, 180.5f),
            Map.of(0, 0.0f, 1, 2.0f, 2, 200.0f));

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      CagraIndexParams indexParams =
          new CagraIndexParams.Builder()
              .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
              .withGraphDegree(1)
              .withIntermediateGraphDegree(2)
              .withNumWriterThreads(4)
              .withMetric(CuvsDistanceType.L2Expanded)
              .build();

      log.trace("Building first index...");
      CagraIndex index1 =
          CagraIndex.newBuilder(resources)
              .withDataset(vector1)
              .withIndexParams(indexParams)
              .build();

      log.trace("Building second index...");
      CagraIndex index2 =
          CagraIndex.newBuilder(resources)
              .withDataset(vector2)
              .withIndexParams(indexParams)
              .build();

      CagraIndexParams outputIndexParams =
          new CagraIndexParams.Builder()
              .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
              .withGraphDegree(2)
              .withIntermediateGraphDegree(4)
              .withNumWriterThreads(4)
              .withMetric(CuvsDistanceType.L2Expanded)
              .build();

      // Host-built indexes are not mergeable. Dim=2 is not 16-byte aligned, so upload to device,
      // allocate owning padded copies, and attach them before merge. Keep them alive until the
      // inputs are closed.
      try (var device1 = CuVSMatrix.ofArray(vector1).toDevice(resources);
          var device2 = CuVSMatrix.ofArray(vector2).toDevice(resources);
          var padded1 = index1.makePaddedDataset(device1);
          var padded2 = index2.makePaddedDataset(device2)) {
        index1.updateDataset(padded1);
        index2.updateDataset(padded2);

        log.trace("Merging indexes with PHYSICAL strategy...");
        try (CagraIndex physicalMergedIndex =
            CagraIndex.merge(new CagraIndex[] {index1, index2}, outputIndexParams)) {
          log.trace("Physical merge completed successfully");

          // Pin SINGLE_CTA; AUTO may pick MULTI_CTA, which drops neighbors on this tiny dataset.
          CagraSearchParams searchParams =
              new CagraSearchParams.Builder()
                  .withAlgo(CagraSearchParams.SearchAlgo.SINGLE_CTA)
                  .build();

          try (var queryVectors = CuVSMatrix.ofArray(queries)) {
            CagraQuery query =
                new CagraQuery.Builder(resources)
                    .withTopK(3)
                    .withSearchParams(searchParams)
                    .withQueryVectors(queryVectors)
                    .withMapping(SearchResults.IDENTITY_MAPPING)
                    .build();

            log.trace("Searching physically merged index...");
            SearchResults physicalResults = physicalMergedIndex.search(query);
            assertNotNull("Physical merge search results should not be null", physicalResults);
            assertEquals(
                "Physical merge search results should match expected",
                expectedResults,
                physicalResults.getResults());

            // --- Serialization/deserialization check for both merged indexes ---
            String physicalIndexFileName = UUID.randomUUID() + ".cag";
            var physicalIndexFile = Path.of(physicalIndexFileName);

            try (var out = Files.newOutputStream(physicalIndexFile)) {
              physicalMergedIndex.serialize(out);
            }

            try (var physicalInputStream = Files.newInputStream(physicalIndexFile);
                CagraIndex loadedPhysicalIndex =
                    CagraIndex.newBuilder(resources).from(physicalInputStream).build()) {

              SearchResults resultsFromLoadedPhysical = loadedPhysicalIndex.search(query);
              assertEquals(
                  "Loaded physical index search results should match expected",
                  expectedResults,
                  resultsFromLoadedPhysical.getResults());
            } finally {
              Files.deleteIfExists(physicalIndexFile);
            }
          }
          index1.close();
          index2.close();
        }
      }
    }
  }
}

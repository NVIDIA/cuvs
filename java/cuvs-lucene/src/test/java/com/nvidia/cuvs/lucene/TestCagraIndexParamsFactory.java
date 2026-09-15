/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;

import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import com.nvidia.cuvs.CagraIndexParams.HnswHeuristicType;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.Test;

/**
 * Verifies that {@link CagraIndexParamsFactory} keeps cuVS as the source of truth for the build
 * heuristics: the GPU-native path defers the build-algorithm choice to cuVS via {@link
 * CagraGraphBuildAlgo#AUTO_SELECT}, and the accelerated-HNSW path defers to cuVS'
 * {@code fromHnswParams} heuristic.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestCagraIndexParamsFactory extends LuceneTestCase {

  /**
   * The GPU-native HEURISTIC path hands the build-algorithm decision to cuVS (AUTO_SELECT) while
   * keeping the caller-supplied CAGRA-native graph degrees, writer threads and metric. Requires the
   * native cuVS library.
   */
  @Test
  public void testGpuHeuristicDelegatesToCuVS() {
    assumeTrue("cuVS not supported", isSupported());

    GPUSearchParams params =
        new GPUSearchParams.Builder()
            .withStrategy(GPUSearchParams.Strategy.HEURISTIC)
            .withGraphDegree(48)
            .withIntermediateGraphDegree(96)
            .withCuvsDistanceType(CuvsDistanceType.InnerProduct)
            .withWriterThreads(4)
            .build();

    // Below cuVS' 1M-row crossover, so the dataset heuristic selects NN-descent.
    CagraIndexParams cagraParams = CagraIndexParamsFactory.create(params, 10_000, 128);

    assertEquals(CagraGraphBuildAlgo.NN_DESCENT, cagraParams.getCagraGraphBuildAlgo());
    // The graph degree is an input to the heuristic, so it survives...
    assertEquals(48, cagraParams.getGraphDegree());
    // ...but the intermediate degree is derived as graph_degree * 3 / 2, overriding the configured
    // 96. cuVS derives the build parameters from that value, so the two must stay in step.
    assertEquals(48 * 3 / 2, cagraParams.getIntermediateGraphDegree());
    // writerThreads has no fromDataset argument and is re-attached by the factory.
    assertEquals(4, cagraParams.getNumWriterThreads());
    assertEquals(CuvsDistanceType.InnerProduct, cagraParams.getCuvsDistanceType());
  }

  /**
   * Build quality reaches cuVS: NN-descent runs {@code 5 + buildQuality} iterations, so two
   * different qualities must produce two different iteration counts. Requires the native cuVS
   * library.
   */
  @Test
  public void testGpuBuildQualityIsHonored() {
    assumeTrue("cuVS not supported", isSupported());

    long low =
        CagraIndexParamsFactory.create(gpuParamsWithQuality(1), 10_000, 128)
            .getNNDescentNumIterations();
    long high =
        CagraIndexParamsFactory.create(gpuParamsWithQuality(15), 10_000, 128)
            .getNNDescentNumIterations();
    long dflt =
        CagraIndexParamsFactory.create(
                gpuParamsWithQuality(GPUSearchParams.DEFAULT_BUILD_QUALITY), 10_000, 128)
            .getNNDescentNumIterations();

    // cuVS derives max_iterations as 5 + buildQuality; assert the relationship it documents rather
    // than hardcoding values it owns.
    assertEquals(low + 14, high);
    assertTrue("higher build quality must not reduce work", dflt > low && dflt < high);
  }

  private static GPUSearchParams gpuParamsWithQuality(int buildQuality) {
    return new GPUSearchParams.Builder()
        .withStrategy(GPUSearchParams.Strategy.HEURISTIC)
        .withBuildQuality(buildQuality)
        .build();
  }

  /**
   * The configured metric must survive every strategy on both paths. It describes the data, not the
   * build strategy: a graph built under the wrong metric degrades recall silently, with no error at
   * build or search time.
   *
   * <p>The accelerated-HNSW HEURISTIC case is the subtle one -- {@code fromHnswParams} forwards the
   * metric to the build heuristic but never assigns it to the params it returns, so reading the
   * metric back off its result yields cuVS' L2Expanded default. Requires the native cuVS library
   * for the paths that call into it.
   */
  @Test
  public void testMetricSurvivesEveryStrategy() {
    assumeTrue("cuVS not supported", isSupported());

    for (GPUSearchParams.Strategy strategy : GPUSearchParams.Strategy.values()) {
      GPUSearchParams gpuParams =
          new GPUSearchParams.Builder()
              .withStrategy(strategy)
              .withCuvsDistanceType(CuvsDistanceType.InnerProduct)
              .build();
      assertEquals(
          "GPU-native path lost the metric under " + strategy,
          CuvsDistanceType.InnerProduct,
          CagraIndexParamsFactory.create(gpuParams, 10_000, 128).getCuvsDistanceType());
    }

    for (AcceleratedHNSWParams.Strategy strategy : AcceleratedHNSWParams.Strategy.values()) {
      AcceleratedHNSWParams hnswParams =
          new AcceleratedHNSWParams.Builder()
              .withStrategy(strategy)
              .withCuvsDistanceType(CuvsDistanceType.InnerProduct)
              .build();
      assertEquals(
          "accelerated-HNSW path lost the metric under " + strategy,
          CuvsDistanceType.InnerProduct,
          CagraIndexParamsFactory.create(hnswParams, 10_000, 128).getCuvsDistanceType());
    }
  }

  /**
   * Only the lower bound is enforced. A negative build quality is rejected at build() time because
   * it would wrap when handed to cuVS' {@code size_t} parameter, but cuVS documents any non-negative
   * value as valid -- so large values must be accepted rather than second-guessed here. Pure Java,
   * no GPU needed.
   */
  @Test
  public void testBuildQualityRejectsNegativeOnly() {
    expectThrows(
        IllegalArgumentException.class,
        () -> new GPUSearchParams.Builder().withBuildQuality(-1).build());

    // Well above the value cuVS calls "most practical"; unusual, but not ours to reject.
    assertEquals(50, new GPUSearchParams.Builder().withBuildQuality(50).build().getBuildQuality());
  }

  /**
   * The GPU-native CUSTOM path passes the explicitly configured build parameters straight through.
   * Pure Java, no GPU needed.
   */
  @Test
  public void testGpuCustomStrategyPassesThroughValues() {
    GPUSearchParams params =
        new GPUSearchParams.Builder()
            .withStrategy(GPUSearchParams.Strategy.CUSTOM)
            .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
            .withGraphDegree(32)
            .withIntermediateGraphDegree(64)
            .withWriterThreads(12)
            .build();

    CagraIndexParams cagraParams = CagraIndexParamsFactory.create(params, 10_000, 128);

    assertEquals(CagraGraphBuildAlgo.NN_DESCENT, cagraParams.getCagraGraphBuildAlgo());
    assertEquals(32, cagraParams.getGraphDegree());
    assertEquals(64, cagraParams.getIntermediateGraphDegree());
    assertNotNull(cagraParams.getCuVSIvfPqParams());
    // The caller-configured writerThreads must be honored on the CUSTOM path.
    assertEquals(12, cagraParams.getNumWriterThreads());
  }

  /**
   * The accelerated-HNSW HEURISTIC path delegates to cuVS' native {@code fromHnswParams}, which
   * derives the graph degrees from maxConn/beamWidth, and re-attaches the caller's writerThreads
   * (which fromHnswParams itself cannot carry). Requires the native cuVS library.
   */
  @Test
  public void testHnswHeuristicDelegatesToCuVS() {
    assumeTrue("cuVS not supported", isSupported());

    AcceleratedHNSWParams params =
        new AcceleratedHNSWParams.Builder()
            .withStrategy(AcceleratedHNSWParams.Strategy.HEURISTIC)
            .withMaxConn(16)
            .withBeamWidth(100)
            .withWriterThreads(7)
            .build();

    CagraIndexParams cagraParams = CagraIndexParamsFactory.create(params, 10_000, 128);

    // SAME_GRAPH_FOOTPRINT yields graph_degree = 2 * maxConn; cuVS owns the exact derivation, so we
    // assert the footprint relationship it documents rather than a hardcoded value.
    assertEquals(2L * params.getMaxConn(), cagraParams.getGraphDegree());
    // The caller-configured writerThreads must be honored on the HEURISTIC path.
    assertEquals(7, cagraParams.getNumWriterThreads());
  }

  /**
   * The heuristic type must reach cuVS rather than being pinned to the default: under
   * SIMILAR_SEARCH_PERFORMANCE cuVS derives {@code graph_degree = 2 + maxConn * 2 / 3} instead of
   * SAME_GRAPH_FOOTPRINT's {@code 2 * maxConn}. Requires the native cuVS library.
   */
  @Test
  public void testHnswHeuristicTypeIsHonored() {
    assumeTrue("cuVS not supported", isSupported());

    AcceleratedHNSWParams params =
        new AcceleratedHNSWParams.Builder()
            .withStrategy(AcceleratedHNSWParams.Strategy.HEURISTIC)
            .withHnswHeuristicType(HnswHeuristicType.SIMILAR_SEARCH_PERFORMANCE)
            .withMaxConn(48)
            .withBeamWidth(100)
            .build();

    assertEquals(HnswHeuristicType.SIMILAR_SEARCH_PERFORMANCE, params.getHnswHeuristicType());

    CagraIndexParams cagraParams = CagraIndexParamsFactory.create(params, 10_000, 128);

    // Distinguishes the two heuristics: SAME_GRAPH_FOOTPRINT would yield 2 * 48 = 96.
    assertEquals(2 + 48 * 2 / 3, cagraParams.getGraphDegree());
  }

  /**
   * The heuristic type defaults to SAME_GRAPH_FOOTPRINT, preserving the behavior callers get without
   * touching the new setter. Pure Java, no GPU needed.
   */
  @Test
  public void testHnswHeuristicTypeDefault() {
    assertEquals(
        AcceleratedHNSWParams.DEFAULT_HNSW_HEURISTIC_TYPE,
        new AcceleratedHNSWParams.Builder().build().getHnswHeuristicType());
  }

  /**
   * A null heuristic type is rejected at build time rather than surfacing as a native failure. Pure
   * Java, no GPU needed.
   */
  @Test
  public void testNullHnswHeuristicTypeRejected() {
    expectThrows(
        IllegalArgumentException.class,
        () -> new AcceleratedHNSWParams.Builder().withHnswHeuristicType(null).build());
  }

  /**
   * Parameters that only the other strategy consumes are still accepted (they are not an error), so
   * that switching strategies does not require rewriting the builder chain -- they are simply not
   * applied. Pure Java, no GPU needed.
   */
  @Test
  public void testStrategySpecificParamsRemainAccepted() {
    AcceleratedHNSWParams hnswParams =
        new AcceleratedHNSWParams.Builder()
            .withStrategy(AcceleratedHNSWParams.Strategy.HEURISTIC)
            .withGraphDegree(96)
            .withIntermediateGraphDegree(192)
            .build();
    // Retained verbatim on the instance; CagraIndexParamsFactory is what declines to apply them.
    assertEquals(96, hnswParams.getGraphdegree());
    assertEquals(192, hnswParams.getIntermediateGraphDegree());

    GPUSearchParams gpuParams =
        new GPUSearchParams.Builder()
            .withStrategy(GPUSearchParams.Strategy.HEURISTIC)
            .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.IVF_PQ)
            .build();
    assertEquals(CagraGraphBuildAlgo.IVF_PQ, gpuParams.getCagraGraphBuildAlgo());
    // ... but for a small dataset cuVS' heuristic selects NN-descent regardless.
    assumeTrue("cuVS not supported", isSupported());
    assertEquals(
        CagraGraphBuildAlgo.NN_DESCENT,
        CagraIndexParamsFactory.create(gpuParams, 10_000, 128).getCagraGraphBuildAlgo());
  }
}

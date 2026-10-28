/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * CAGRA indices built from BBQ quantizers via {@link CagraIndex.Builder#fromBbq(BbqQuantizer)}.
 *
 * <p>{@code gpuSearchBaselineWithoutBbq} is a control: the same search over a graph plain CAGRA
 * built from the same vectors. If it regresses alongside the BBQ case, look at the search
 * parameters or the data before suspecting quantization.
 */
public class CagraBbqIT extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

  private static final int ROWS = 20000;
  private static final int DIM = 64;
  private static final int QUERIES = 32;
  private static final int CLUSTERS = 40;
  private static final double CLUSTER_SPREAD = 3.0;

  /** A neighbor within this much of the true k-th distance is as good an answer. */
  private static final float TIE_TOLERANCE = 0.01f;

  private static final int TOP_K = 5;

  /**
   * Floor for recall@{@value #TOP_K}. Both searches measure 1.0 on this data with the pinned search
   * parameters, so this leaves room for nn-descent's run-to-run variation without being loose
   * enough to pass a real regression.
   */
  private static final double RECALL_FLOOR = 0.95;

  /** PACKED_1B encodes one bit per dimension. */
  private static final int CODE_BYTES = (DIM + 7) / 8;

  private float[][] dataset;
  private float[][] queries;

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxSupportedArch());
    // Fixed seed so the dataset and queries are stable. Note this does not make recall stable:
    // nn-descent is stochastic, so the measured values still move by a few points between runs.
    random = new java.util.Random(42);
    float[][] centers = randomVectors(CLUSTERS);
    dataset = clusteredVectors(centers, ROWS);
    queries = clusteredVectors(centers, QUERIES);
  }

  /** Builds the graph from quantizers with a dense dataset attached, then searches on the GPU. */
  @Test
  public void gpuBuildGpuSearch() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create();
        Quantizer quantizer = newQuantizer(resources)) {
      // Owned by the index once built, so it is not closed here.
      CuVSMatrix dense = newDenseDataset(resources);
      try (CagraIndex index = bbqIndex(resources, quantizer, dense);
          CuVSMatrix queryVectors = CuVSMatrix.ofArray(queries)) {
        CagraQuery query =
            new CagraQuery.Builder(resources)
                .withTopK(TOP_K)
                .withSearchParams(searchParams())
                .withQueryVectors(queryVectors)
                .withMapping(SearchResults.IDENTITY_MAPPING)
                .build();
        assertRecall("gpu-search", index.search(query));
      }
    }
  }

  /**
   * Plain CAGRA over the same vectors, as a reference point for the BBQ recall above. Both sides
   * run the same GPU search, so the difference between the two numbers is the cost of building the
   * graph from quantized codes.
   */
  @Test
  public void gpuSearchBaselineWithoutBbq() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create()) {
      // Owned by the index once built, so it is not closed here.
      CuVSMatrix dense = newDenseDataset(resources);
      try (CagraIndex index =
              CagraIndex.newBuilder(resources)
                  .fromDataset(dense)
                  .withIndexParams(params())
                  .build();
          CuVSMatrix queryVectors = CuVSMatrix.ofArray(queries)) {
        CagraQuery query =
            new CagraQuery.Builder(resources)
                .withTopK(TOP_K)
                .withSearchParams(searchParams())
                .withQueryVectors(queryVectors)
                .withMapping(SearchResults.IDENTITY_MAPPING)
                .build();
        assertRecall("baseline-gpu-search", index.search(query));
      }
    }
  }

  /**
   * Extracting the graph from a graph-only BBQ index, which is how cuvs-lucene consumes one: no
   * dense dataset is attached, so this is the memory-efficient shape BBQ exists for, and nothing
   * native dereferences the BBQ dataset view on this path.
   */
  @Test
  public void graphOnlyBbqBuildProducesUsableGraph() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create();
        Quantizer quantizer = newQuantizer(resources)) {
      try (CagraIndex index = bbqIndex(resources, quantizer, null);
          CuVSMatrix graph = index.getGraph()) {
        assertEquals("one adjacency row per vector", ROWS, graph.size());
        int degree = (int) graph.columns();
        assertTrue("graph degree should be positive", degree > 0);

        int[][] adjacency = readAdjacency(graph);
        int selfLoops = 0;
        for (int node = 0; node < ROWS; node++) {
          for (int neighbor : adjacency[node]) {
            assertTrue(
                "neighbor id " + neighbor + " of node " + node + " is out of range",
                neighbor >= 0 && neighbor < ROWS);
            if (neighbor == node) {
              selfLoops++;
            }
          }
        }
        // A handful of self-references is normal padding; a graph made mostly of them is not.
        assertTrue(
            "graph is mostly self-references (" + selfLoops + " of " + ROWS * degree + ")",
            selfLoops < ROWS * degree / 10);
      }
    }
  }

  /**
   * Without parameters a BBQ build uses the quantizers' metric, not the L2Expanded default of
   * {@link CagraIndexParams.Builder}: nn-descent requires its metric to match the quantizers', so
   * an inner-product quantizer only builds if the metric was carried over. The rest of the
   * parameters keep their defaults, which shows in the graph degree.
   */
  @Test
  public void buildsWithTheQuantizersMetricByDefault() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create();
        Quantizer quantizer = newQuantizer(resources, CuvsDistanceType.InnerProduct);
        CagraIndex index = CagraIndex.newBuilder(resources).fromBbq(quantizer.quantizer()).build();
        CuVSMatrix graph = index.getGraph()) {
      assertEquals("one adjacency row per vector", ROWS, graph.size());
      assertEquals(
          "the default graph degree",
          new CagraIndexParams.Builder().build().getGraphDegree(),
          graph.columns());
      for (int[] neighbors : readAdjacency(graph)) {
        for (int neighbor : neighbors) {
          assertTrue(
              "neighbor id " + neighbor + " is out of range", neighbor >= 0 && neighbor < ROWS);
        }
      }
    }
  }

  /**
   * A dense dataset handed to the builder is owned by the resulting index, exactly as it is for a
   * non-BBQ build, so closing the index releases it. Host matrices are arena-backed, which makes
   * the release observable: reads fail once the arena is gone.
   */
  @Test
  public void indexOwnsTheDenseDatasetItWasGiven() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create();
        Quantizer quantizer = newQuantizer(resources)) {
      // hostBuilder yields an arena-backed matrix; ofArray's close() is a no-op and would make
      // this test pass whether or not the index released anything.
      var denseBuilder = CuVSMatrix.hostBuilder(ROWS, DIM, CuVSMatrix.DataType.FLOAT);
      for (float[] row : dataset) {
        denseBuilder.addVector(row);
      }
      CuVSMatrix dense = denseBuilder.build();
      try (CagraIndex index = bbqIndex(resources, quantizer, dense)) {
        assertNotNull("the dataset should be readable while the index is open", dense.getRow(0));
      }
      // The read, not getRow itself, is what validates the arena's scope.
      assertThrows(
          "closing the index should have released the dataset it was given",
          IllegalStateException.class,
          () -> dense.getRow(0).getAsFloat(0));
    }
  }

  /**
   * Every quantizer component is required. {@code centroidNormSq} is the one that cannot rely on a
   * null check, since it is a primitive whose unset state would otherwise read as a legitimate
   * zero and quietly skew inner-product and cosine distances.
   */
  @Test
  public void quantizerMustRejectMissingCentroidNormSq() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create();
        Quantizer quantizer = newQuantizer(resources)) {
      var q = quantizer.quantizer();
      assertThrows(
          "omitting withCentroidNormSq should fail fast, like every other component does",
          RuntimeException.class,
          () ->
              new BbqQuantizer.Builder()
                  .withCodes(q.getCodes())
                  .withLowerIntervals(q.getLowerIntervals())
                  .withUpperIntervals(q.getUpperIntervals())
                  .withAdditionalCorrections(q.getAdditionalCorrections())
                  .withQuantizedComponentSums(q.getQuantizedComponentSums())
                  .withCentroid(q.getCentroid())
                  .withDequantDelta(q.getDequantDelta())
                  .withDequantSumDelta(q.getDequantSumDelta())
                  .withRowNorm(q.getRowNorm())
                  .withLayout(BbqQuantizer.CodeLayout.PACKED_1B)
                  .withMetric(CuvsDistanceType.L2Expanded)
                  // withCentroidNormSq deliberately omitted
                  .build());
    }
  }

  /**
   * Two encodings of the same vectors build in either order, because cuVS picks the role of each
   * one from its layout. The 2-bit encoding decodes to the same values as the 1-bit one, so the
   * asymmetric distances equal the symmetric 1-bit ones and recall is held to the same floor.
   */
  @Test
  public void asymmetricPairBuildsInEitherOrder() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create();
        Quantizer oneBit = newQuantizer(resources);
        Quantizer twoBit = newTransposed2bQuantizer(resources);
        CuVSMatrix queryVectors = CuVSMatrix.ofArray(queries)) {
      for (boolean oneBitFirst : new boolean[] {true, false}) {
        var first = oneBitFirst ? oneBit : twoBit;
        var second = oneBitFirst ? twoBit : oneBit;
        // Owned by the index once built, so it is not closed here.
        CuVSMatrix dense = newDenseDataset(resources);
        try (CagraIndex index =
            CagraIndex.newBuilder(resources)
                .fromBbq(first.quantizer(), second.quantizer())
                .withDenseDataset(dense)
                .withIndexParams(params())
                .build()) {
          CagraQuery query =
              new CagraQuery.Builder(resources)
                  .withTopK(TOP_K)
                  .withSearchParams(searchParams())
                  .withQueryVectors(queryVectors)
                  .withMapping(SearchResults.IDENTITY_MAPPING)
                  .build();
          assertRecall(
              first.quantizer().getLayout() + " + " + second.quantizer().getLayout(),
              index.search(query));
        }
      }
    }
  }

  /**
   * The deprecated varargs setter is the only way left to pass a number of quantizers other than
   * one or two, so the check behind it stays covered until the setter is removed.
   */
  @Test
  @SuppressWarnings("removal")
  public void deprecatedSetterRejectsAnUnusableQuantizerCount() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create();
        Quantizer quantizer = newQuantizer(resources)) {
      var q = quantizer.quantizer();
      assertThrows(
          "a build needs at least one representation",
          IllegalArgumentException.class,
          () -> CagraIndex.newBuilder(resources).withBbqDataset().build());
      assertThrows(
          "three representations are more than the native side accepts",
          IllegalArgumentException.class,
          () -> CagraIndex.newBuilder(resources).withBbqDataset(q, q, q).build());
      assertThrows(
          "a null representation should be caught before it reaches native code",
          NullPointerException.class,
          () -> CagraIndex.newBuilder(resources).withBbqDataset(q, null).build());
    }
  }

  /** Quantizer components live on the device; a host matrix cannot be read by the native side. */
  @Test
  public void hostQuantizerComponentIsRejected() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create();
        Quantizer quantizer = newQuantizer(resources)) {
      var q = quantizer.quantizer();
      var hostBuilder = CuVSMatrix.hostBuilder(ROWS, 1, CuVSMatrix.DataType.FLOAT);
      for (int i = 0; i < ROWS; i++) {
        hostBuilder.addVector(new float[] {1.0f});
      }
      try (CuVSMatrix hostRowNorm = hostBuilder.build()) {
        var withHostComponent =
            new BbqQuantizer.Builder()
                .withCodes(q.getCodes())
                .withLowerIntervals(q.getLowerIntervals())
                .withUpperIntervals(q.getUpperIntervals())
                .withAdditionalCorrections(q.getAdditionalCorrections())
                .withQuantizedComponentSums(q.getQuantizedComponentSums())
                .withCentroid(q.getCentroid())
                .withDequantDelta(q.getDequantDelta())
                .withDequantSumDelta(q.getDequantSumDelta())
                .withRowNorm(hostRowNorm)
                .withLayout(BbqQuantizer.CodeLayout.PACKED_1B)
                .withMetric(CuvsDistanceType.L2Expanded)
                .withCentroidNormSq(q.getCentroidNormSq())
                .build();
        var failure =
            assertThrows(
                IllegalArgumentException.class, () -> bbqIndex(resources, withHostComponent, null));
        assertTrue(
            "the message should name the offending component, but was: " + failure.getMessage(),
            failure.getMessage().contains("rowNorm"));
      }
    }
  }

  /** A graph-only index carries no vectors, so searching it before attaching a dataset fails. */
  @Test
  public void graphOnlyIndexCannotBeSearchedUntilADatasetIsAttached() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create();
        Quantizer quantizer = newQuantizer(resources)) {
      try (CagraIndex index = bbqIndex(resources, quantizer, null);
          CuVSMatrix queryVectors = CuVSMatrix.ofArray(queries)) {
        CagraQuery query =
            new CagraQuery.Builder(resources)
                .withTopK(TOP_K)
                .withSearchParams(searchParams())
                .withQueryVectors(queryVectors)
                .withMapping(SearchResults.IDENTITY_MAPPING)
                .build();
        var failure = assertThrows(RuntimeException.class, () -> index.search(query));
        assertTrue(
            "the message should point at updateDataset, but was: " + failure.getMessage(),
            failure.getMessage().contains("cuvsCagraUpdateDataset"));
      }
    }
  }

  /** The documented follow-up: attach a dataset to a graph-only index and it becomes searchable. */
  @Test
  public void attachingADatasetMakesAGraphOnlyIndexSearchable() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create();
        Quantizer quantizer = newQuantizer(resources);
        CuVSMatrix dense = newDenseDataset(resources)) {
      try (CagraIndex index = bbqIndex(resources, quantizer, null);
          CuVSMatrix queryVectors = CuVSMatrix.ofArray(queries)) {
        try (CagraIndex.PaddedDatasetView padded = index.makePaddedDatasetView(dense)) {
          index.updateDataset(padded);
          CagraQuery query =
              new CagraQuery.Builder(resources)
                  .withTopK(TOP_K)
                  .withSearchParams(searchParams())
                  .withQueryVectors(queryVectors)
                  .withMapping(SearchResults.IDENTITY_MAPPING)
                  .build();
          var results = index.search(query).getResults();
          assertEquals("one result set per query", QUERIES, results.size());
          for (var perQuery : results) {
            assertEquals("each query should return topK neighbors", TOP_K, perQuery.size());
          }
        }
      }
    }
  }

  /**
   * A component may declare a row stride explicitly as long as it equals the column count. That is
   * still contiguous, so it has to be accepted and passed to the native side without strides.
   */
  @Test
  public void explicitlyContiguousCodesMatrixIsAccepted() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create();
        Quantizer quantizer = newQuantizer(resources)) {
      var q = quantizer.quantizer();
      var codesBuilder =
          CuVSMatrix.deviceBuilder(
              resources, ROWS, CODE_BYTES, CODE_BYTES, -1, CuVSMatrix.DataType.BYTE);
      for (int i = 0; i < ROWS; i++) {
        codesBuilder.addVector(new byte[CODE_BYTES]);
      }
      try (CuVSMatrix stridedCodes = codesBuilder.build()) {
        var withStridedCodes =
            new BbqQuantizer.Builder()
                .withCodes(stridedCodes)
                .withLowerIntervals(q.getLowerIntervals())
                .withUpperIntervals(q.getUpperIntervals())
                .withAdditionalCorrections(q.getAdditionalCorrections())
                .withQuantizedComponentSums(q.getQuantizedComponentSums())
                .withCentroid(q.getCentroid())
                .withDequantDelta(q.getDequantDelta())
                .withDequantSumDelta(q.getDequantSumDelta())
                .withRowNorm(q.getRowNorm())
                .withLayout(BbqQuantizer.CodeLayout.PACKED_1B)
                .withMetric(CuvsDistanceType.L2Expanded)
                .withCentroidNormSq(q.getCentroidNormSq())
                .build();
        try (CagraIndex index = bbqIndex(resources, withStridedCodes, null);
            CuVSMatrix graph = index.getGraph()) {
          assertEquals("one adjacency row per vector", ROWS, graph.size());
        }
      }
    }
  }

  /**
   * Every quantizer component has to be contiguous, because the native side reads them through
   * {@code from_dlpack}, which accepts only compact row-major memory. The check belongs on the Java
   * side so the message can name the offending tensor rather than just the entry point.
   */
  @Test
  public void paddedCodesMatrixMustBeRejectedClearly() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create();
        Quantizer quantizer = newQuantizer(resources)) {
      var q = quantizer.quantizer();
      var paddedCodesBuilder =
          CuVSMatrix.deviceBuilder(
              resources, ROWS, CODE_BYTES, CODE_BYTES * 2, -1, CuVSMatrix.DataType.BYTE);
      for (int i = 0; i < ROWS; i++) {
        paddedCodesBuilder.addVector(new byte[CODE_BYTES]);
      }
      try (CuVSMatrix paddedCodes = paddedCodesBuilder.build()) {
        var padded =
            new BbqQuantizer.Builder()
                .withCodes(paddedCodes)
                .withLowerIntervals(q.getLowerIntervals())
                .withUpperIntervals(q.getUpperIntervals())
                .withAdditionalCorrections(q.getAdditionalCorrections())
                .withQuantizedComponentSums(q.getQuantizedComponentSums())
                .withCentroid(q.getCentroid())
                .withDequantDelta(q.getDequantDelta())
                .withDequantSumDelta(q.getDequantSumDelta())
                .withRowNorm(q.getRowNorm())
                .withLayout(BbqQuantizer.CodeLayout.PACKED_1B)
                .withMetric(CuvsDistanceType.L2Expanded)
                .withCentroidNormSq(q.getCentroidNormSq())
                .build();
        var failure =
            assertThrows(
                "a padded codes matrix should be rejected by the Java guard, as vector tensors are",
                IllegalArgumentException.class,
                () ->
                    CagraIndex.newBuilder(resources)
                        .fromBbq(padded)
                        .withIndexParams(
                            new CagraIndexParams.Builder()
                                .withMetric(CuvsDistanceType.L2Expanded)
                                .build())
                        .build());
        assertTrue(
            "the message should name the offending component, but was: " + failure.getMessage(),
            failure.getMessage().contains("codes"));
      }
    }
  }

  private static int[][] readAdjacency(CuVSMatrix graph) {
    int nodes = (int) graph.size();
    int degree = (int) graph.columns();
    int[][] adjacency = new int[nodes][degree];
    for (int i = 0; i < nodes; i++) {
      RowView row = graph.getRow(i);
      for (int j = 0; j < degree; j++) {
        adjacency[i][j] = row.getAsInt(j);
      }
    }
    return adjacency;
  }

  /**
   * Recall@{@value #TOP_K} against brute-force ground truth, scored by distance rather than by id.
   *
   * <p>A returned neighbor counts when it is at least as close as the true {@value #TOP_K}-th, so
   * an answer that is equally good but differently tied is not penalised. Scoring by id instead
   * measures how the data happens to break ties, which on synthetic vectors is mostly noise.
   */
  private void assertRecall(String label, SearchResults results) {
    int hits = 0;
    int total = 0;
    for (int q = 0; q < QUERIES; q++) {
      float[] distances = new float[ROWS];
      for (int i = 0; i < ROWS; i++) {
        distances[i] = squaredDistance(queries[q], dataset[i]);
      }
      float[] sorted = distances.clone();
      Arrays.sort(sorted);
      float acceptable = sorted[TOP_K - 1] * (1.0f + TIE_TOLERANCE);
      for (int id : results.getResults().get(q).keySet()) {
        total++;
        // CAGRA reports a missing neighbor as an out-of-range id; count it as a miss rather than
        // indexing with it.
        if (id >= 0 && id < ROWS && distances[id] <= acceptable) {
          hits++;
        }
      }
    }
    double recall = (double) hits / total;
    log.info("{} recall@{} = {}", label, TOP_K, recall);
    assertTrue(label + " recall@" + TOP_K + " was only " + recall, recall >= RECALL_FLOOR);
  }

  /**
   * Search effort is pinned rather than left to the defaults: recall is being asserted here, and a
   * default that changes would silently move the number the assertion is checking.
   */
  private static CagraSearchParams searchParams() {
    return new CagraSearchParams.Builder()
        .withAlgo(CagraSearchParams.SearchAlgo.SINGLE_CTA)
        .withItopkSize(256)
        .build();
  }

  private static float squaredDistance(float[] a, float[] b) {
    float sum = 0.0f;
    for (int i = 0; i < a.length; i++) {
      float diff = a[i] - b[i];
      sum += diff * diff;
    }
    return sum;
  }

  private static CagraIndex bbqIndex(
      CuVSResources resources, Quantizer quantizer, CuVSMatrix dataset) throws Throwable {
    return bbqIndex(resources, quantizer.quantizer(), dataset);
  }

  private static CagraIndex bbqIndex(
      CuVSResources resources, BbqQuantizer quantizer, CuVSMatrix dataset) throws Throwable {
    var builder = CagraIndex.newBuilder(resources).fromBbq(quantizer).withIndexParams(params());
    if (dataset != null) {
      builder.withDenseDataset(dataset);
    }
    return builder.build();
  }

  /** The same graph shape for every build here, with and without BBQ. */
  private static CagraIndexParams params() {
    return new CagraIndexParams.Builder()
        .withMetric(CuvsDistanceType.L2Expanded)
        .withGraphDegree(32)
        .withIntermediateGraphDegree(64)
        .build();
  }

  private float[][] randomVectors(int count) {
    float[][] vectors = new float[count][DIM];
    for (float[] vector : vectors) {
      for (int d = 0; d < DIM; d++) {
        vector[d] = random.nextFloat() * 100.0f;
      }
    }
    return vectors;
  }

  /**
   * Vectors drawn around a set of cluster centers.
   *
   * <p>Uniformly random high-dimensional vectors are close to equidistant from one another, so a
   * graph index has nothing to exploit and recall says more about the data than the
   * implementation. Clustered data is both more representative and a meaningful thing to assert on.
   */
  private float[][] clusteredVectors(float[][] centers, int count) {
    float[][] vectors = new float[count][DIM];
    for (float[] vector : vectors) {
      float[] center = centers[random.nextInt(centers.length)];
      for (int d = 0; d < DIM; d++) {
        vector[d] = center[d] + (float) (random.nextGaussian() * CLUSTER_SPREAD);
      }
    }
    return vectors;
  }

  private CuVSMatrix newDenseDataset(CuVSResources resources) {
    var builder = CuVSMatrix.deviceBuilder(resources, ROWS, DIM, CuVSMatrix.DataType.FLOAT);
    for (float[] row : dataset) {
      builder.addVector(row);
    }
    return builder.build();
  }

  /** A quantizer plus the component matrices it was built from, closed together. */
  private record Quantizer(BbqQuantizer quantizer, List<CuVSMatrix> matrices)
      implements AutoCloseable {
    @Override
    public void close() {
      matrices.forEach(CuVSMatrix::close);
    }
  }

  /**
   * Encodes {@link #dataset} as PACKED_1B BBQ codes and uploads every component to device memory.
   *
   * <p>This mirrors {@code cpp/internal/cuvs_internal/preprocessing/bbq_cpu_quantize.hpp} closely
   * enough for the codes to describe the real data, which is what makes the end-to-end searches
   * below meaningful. It deliberately skips that file's {@code optimize_intervals} MSE refinement
   * and keeps the initial grid estimate: the graph only has to be good, not optimal, because both
   * searches score candidates against the dense dataset rather than against the codes.
   */
  private Quantizer newQuantizer(CuVSResources resources) {
    return newQuantizer(resources, CuvsDistanceType.L2Expanded);
  }

  private Quantizer newQuantizer(CuVSResources resources, CuvsDistanceType metric) {
    OneBitCodes oneBit = encodeOneBit(metric);
    float[] delta = new float[ROWS];
    float[] sumDelta = new float[ROWS];
    for (int i = 0; i < ROWS; i++) {
      // One step for 1-bit codes, so each component lands on 0 or 1.
      delta[i] = oneBit.upper()[i] - oneBit.lower()[i];
      sumDelta[i] = delta[i] * oneBit.sums()[i];
    }
    return upload(
        resources,
        BbqQuantizer.CodeLayout.PACKED_1B,
        oneBit.codes(),
        oneBit.sums(),
        delta,
        sumDelta,
        oneBit);
  }

  /**
   * A TRANSPOSED_2B encoding of {@link #dataset}, derived from the 1-bit one instead of quantized
   * afresh. On the same interval, the 2-bit code {@code 3c} decodes to exactly the value the 1-bit
   * code {@code c} does, because the 2-bit step is a third of the 1-bit one. Both bit planes of
   * {@code 3c} equal {@code c}, so each row is the 1-bit row twice, plane after plane, as
   * {@code pack_codes} in the C++ reference lays them out.
   */
  private Quantizer newTransposed2bQuantizer(CuVSResources resources) {
    OneBitCodes oneBit = encodeOneBit(CuvsDistanceType.L2Expanded);
    byte[][] codes = new byte[ROWS][2 * CODE_BYTES];
    int[] sums = new int[ROWS];
    float[] delta = new float[ROWS];
    float[] sumDelta = new float[ROWS];
    for (int i = 0; i < ROWS; i++) {
      System.arraycopy(oneBit.codes()[i], 0, codes[i], 0, CODE_BYTES);
      System.arraycopy(oneBit.codes()[i], 0, codes[i], CODE_BYTES, CODE_BYTES);
      sums[i] = 3 * oneBit.sums()[i];
      delta[i] = (oneBit.upper()[i] - oneBit.lower()[i]) / 3;
      sumDelta[i] = delta[i] * sums[i];
    }
    return upload(
        resources, BbqQuantizer.CodeLayout.TRANSPOSED_2B, codes, sums, delta, sumDelta, oneBit);
  }

  /** The host-side result of {@link #encodeOneBit(CuvsDistanceType)}. */
  private record OneBitCodes(
      CuvsDistanceType metric,
      byte[][] codes,
      float[] lower,
      float[] upper,
      float[] corrections,
      int[] sums,
      float[] rowNorms,
      float[] centroid,
      float centroidNormSq) {}

  private OneBitCodes encodeOneBit(CuvsDistanceType metric) {
    boolean euclidean =
        metric == CuvsDistanceType.L2Expanded || metric == CuvsDistanceType.L2SqrtExpanded;
    float[] centroid = new float[DIM];
    for (float[] row : dataset) {
      for (int d = 0; d < DIM; d++) {
        centroid[d] += row[d];
      }
    }
    float centroidNormSq = 0.0f;
    for (int d = 0; d < DIM; d++) {
      centroid[d] /= ROWS;
      centroidNormSq += centroid[d] * centroid[d];
    }

    byte[][] codes = new byte[ROWS][CODE_BYTES];
    float[] lower = new float[ROWS];
    float[] upper = new float[ROWS];
    float[] corrections = new float[ROWS];
    int[] sums = new int[ROWS];
    float[] rowNorms = new float[ROWS];

    for (int i = 0; i < ROWS; i++) {
      float[] centred = new float[DIM];
      float origNormSq = 0.0f;
      float min = Float.MAX_VALUE;
      float max = -Float.MAX_VALUE;
      float centredNormSq = 0.0f;
      float centroidDot = 0.0f;
      double mean = 0.0;
      double var = 0.0;
      for (int d = 0; d < DIM; d++) {
        origNormSq += dataset[i][d] * dataset[i][d];
        centroidDot += dataset[i][d] * centroid[d];
        centred[d] = dataset[i][d] - centroid[d];
        min = Math.min(min, centred[d]);
        max = Math.max(max, centred[d]);
        centredNormSq += centred[d] * centred[d];
        double diff = centred[d] - mean;
        mean += diff / (d + 1);
        var += diff * (centred[d] - mean);
      }
      double stddev = Math.sqrt(var / DIM);

      // kMinimumMseGrid[0] from the C++ reference, i.e. the 1-bit row.
      float a = (float) clamp(-0.798 * stddev + mean, min, max);
      float b = (float) clamp(0.798 * stddev + mean, min, max);
      float step = b - a;
      int sum = 0;
      for (int d = 0; d < DIM; d++) {
        int code = step == 0.0f ? 0 : Math.round((float) (clamp(centred[d], a, b) - a) / step);
        sum += code;
        codes[i][d / 8] |= (byte) ((code & 1) << (7 - (d % 8)));
      }

      lower[i] = a;
      upper[i] = b;
      // As in the C++ reference: the centred norm for a euclidean metric, otherwise the original
      // vector's dot product with the centroid.
      corrections[i] = euclidean ? centredNormSq : centroidDot;
      sums[i] = sum;
      rowNorms[i] = origNormSq;
    }
    return new OneBitCodes(
        metric, codes, lower, upper, corrections, sums, rowNorms, centroid, centroidNormSq);
  }

  /** Uploads one encoding to device memory; the components that don't depend on it come shared. */
  private static Quantizer upload(
      CuVSResources resources,
      BbqQuantizer.CodeLayout layout,
      byte[][] codes,
      int[] sums,
      float[] delta,
      float[] sumDelta,
      OneBitCodes shared) {
    List<CuVSMatrix> owned = new ArrayList<>();
    var codesBuilder =
        CuVSMatrix.deviceBuilder(resources, ROWS, codes[0].length, CuVSMatrix.DataType.BYTE);
    for (byte[] row : codes) {
      codesBuilder.addVector(row);
    }

    BbqQuantizer quantizer =
        new BbqQuantizer.Builder()
            .withCodes(track(owned, codesBuilder.build()))
            .withLowerIntervals(track(owned, deviceVector(resources, shared.lower())))
            .withUpperIntervals(track(owned, deviceVector(resources, shared.upper())))
            .withAdditionalCorrections(track(owned, deviceVector(resources, shared.corrections())))
            .withQuantizedComponentSums(track(owned, deviceVector(resources, sums)))
            .withCentroid(track(owned, deviceRow(resources, shared.centroid())))
            .withDequantDelta(track(owned, deviceVector(resources, delta)))
            .withDequantSumDelta(track(owned, deviceVector(resources, sumDelta)))
            .withRowNorm(track(owned, deviceVector(resources, shared.rowNorms())))
            .withLayout(layout)
            .withMetric(shared.metric())
            .withCentroidNormSq(shared.centroidNormSq())
            .build();
    return new Quantizer(quantizer, owned);
  }

  private static double clamp(double value, double min, double max) {
    return Math.min(Math.max(value, min), max);
  }

  private static CuVSMatrix track(List<CuVSMatrix> owned, CuVSMatrix matrix) {
    owned.add(matrix);
    return matrix;
  }

  /** One value per row, which the BBQ view reads as a length-ROWS vector. */
  private static CuVSMatrix deviceVector(CuVSResources resources, float[] values) {
    var builder = CuVSMatrix.deviceBuilder(resources, values.length, 1, CuVSMatrix.DataType.FLOAT);
    for (float value : values) {
      builder.addVector(new float[] {value});
    }
    return builder.build();
  }

  private static CuVSMatrix deviceVector(CuVSResources resources, int[] values) {
    var builder = CuVSMatrix.deviceBuilder(resources, values.length, 1, CuVSMatrix.DataType.INT);
    for (int value : values) {
      builder.addVector(new int[] {value});
    }
    return builder.build();
  }

  private static CuVSMatrix deviceRow(CuVSResources resources, float[] row) {
    var builder = CuVSMatrix.deviceBuilder(resources, 1, row.length, CuVSMatrix.DataType.FLOAT);
    builder.addVector(row);
    return builder.build();
  }
}

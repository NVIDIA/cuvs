/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import com.carrotsearch.randomizedtesting.RandomizedRunner;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.lang.reflect.Proxy;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/**
 * The checks of the builders returned by {@link CagraIndex.Builder}'s {@code from*} methods. Most
 * of them fail before anything reaches native code; building, searching and loading are covered by
 * the other CAGRA tests.
 */
@RunWith(RandomizedRunner.class)
public class CagraIndexBuilderIT extends CuVSTestCase {

  private static final int ROWS = 256;
  private static final int DIM = 16;

  private float[][] vectors;

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxSupportedArch());
    initializeRandom();
    vectors = generateData(random, ROWS, DIM);
  }

  @Test
  public void aBuilderBuildsOnce() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create()) {
      var datasetBuilder = CagraIndex.newBuilder(resources).fromDataset(vectors);
      try (CagraIndex index = datasetBuilder.build()) {
        assertEquals(ROWS, index.size());
      }
      assertThrows(IllegalStateException.class, datasetBuilder::build);
    }
  }

  @Test
  public void aBuilderStartsOneIndex() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create()) {
      var builder = CagraIndex.newBuilder(resources);
      builder.fromDataset(vectors);
      var failure =
          assertThrows(
              IllegalStateException.class,
              () -> builder.fromSerialized(new ByteArrayInputStream(new byte[0])));
      assertTrue(failure.getMessage(), failure.getMessage().contains("fromDataset"));
    }
  }

  @Test
  public void fromBbqRejectsTwoQuantizersWithTheSameLayout() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create()) {
      var builder = CagraIndex.newBuilder(resources);
      var failure =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  builder.fromBbq(
                      quantizer(BbqQuantizer.CodeLayout.PACKED_1B),
                      quantizer(BbqQuantizer.CodeLayout.PACKED_1B)));
      assertTrue(failure.getMessage(), failure.getMessage().contains("PACKED_1B"));
      // The rejected call didn't use the builder up.
      builder.fromDataset(vectors);
    }
  }

  @Test
  public void fromBbqRejectsParametersForAnotherGraphBuildAlgorithm() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create()) {
      var bbqBuilder =
          CagraIndex.newBuilder(resources).fromBbq(quantizer(BbqQuantizer.CodeLayout.PACKED_1B));
      var ivfPq =
          new CagraIndexParams.Builder()
              .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.IVF_PQ)
              .build();
      var failure =
          assertThrows(IllegalArgumentException.class, () -> bbqBuilder.withIndexParams(ivfPq));
      assertTrue(failure.getMessage(), failure.getMessage().contains("IVF_PQ"));
    }
  }

  @Test
  public void rejectsAnInputGivenTwice() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create()) {
      var params = new CagraIndexParams.Builder().build();
      var datasetBuilder = CagraIndex.newBuilder(resources).fromDataset(vectors);
      datasetBuilder.withIndexParams(params);
      assertThrows(IllegalStateException.class, () -> datasetBuilder.withIndexParams(params));

      var bbqBuilder =
          CagraIndex.newBuilder(resources).fromBbq(quantizer(BbqQuantizer.CodeLayout.PACKED_1B));
      var dense = fake(CuVSMatrix.class);
      bbqBuilder.withDenseDataset(dense).withIndexParams(params);
      assertThrows(IllegalStateException.class, () -> bbqBuilder.withDenseDataset(dense));
      assertThrows(IllegalStateException.class, () -> bbqBuilder.withIndexParams(params));

      var serializedBuilder =
          CagraIndex.newBuilder(resources).fromSerialized(new ByteArrayInputStream(new byte[0]));
      serializedBuilder.withOutputDataset(new CagraIndex.StandardDataset());
      var failure =
          assertThrows(
              IllegalStateException.class,
              () -> serializedBuilder.withOutputDataset(new CagraIndex.StandardDataset()));
      assertTrue(failure.getMessage(), failure.getMessage().startsWith("withOutputDataset"));
    }
  }

  @Test
  public void fromBbqTakesItsMetricFromTheQuantizers() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create()) {
      var quantizer = quantizer(BbqQuantizer.CodeLayout.PACKED_1B); // encoded for L2Expanded
      var innerProduct =
          new CagraIndexParams.Builder().withMetric(CuvsDistanceType.InnerProduct).build();
      var failure =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  CagraIndex.newBuilder(resources)
                      .fromBbq(quantizer)
                      .withIndexParams(innerProduct));
      assertTrue(failure.getMessage(), failure.getMessage().contains("InnerProduct"));

      // AUTO_SELECT picks nn-descent for a BBQ dataset, so it is accepted.
      CagraIndex.newBuilder(resources)
          .fromBbq(quantizer)
          .withIndexParams(
              new CagraIndexParams.Builder()
                  .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.AUTO_SELECT)
                  .build());
    }
  }

  @Test
  public void fromGraphNeedsADeviceDataset() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create()) {
      var hostMatrix = fake(CuVSMatrix.class);
      var failure =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  CagraIndex.newBuilder(resources)
                      .fromGraph(CuvsDistanceType.L2Expanded, hostMatrix, hostMatrix));
      assertTrue(failure.getMessage(), failure.getMessage().contains("device memory"));
    }
  }

  /** The stream belongs to the caller, so loading leaves it open, whether or not it succeeds. */
  @Test
  public void fromSerializedLeavesTheStreamOpen() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create()) {
      byte[] serialized;
      try (CagraIndex index = CagraIndex.newBuilder(resources).fromDataset(vectors).build()) {
        var bytes = new ByteArrayOutputStream();
        index.serialize(bytes);
        serialized = bytes.toByteArray();
      }

      var stream = new CloseTrackingStream(serialized);
      try (CagraIndex loaded = CagraIndex.newBuilder(resources).fromSerialized(stream).build()) {
        assertEquals(ROWS, loaded.size());
      }
      assertFalse(stream.closed);

      var failing = new CloseTrackingStream(serialized);
      try (var notEmpty = new CagraIndex.StandardDataset()) {
        notEmpty.setDelegate(() -> {}, 1);
        var serializedBuilder =
            CagraIndex.newBuilder(resources).fromSerialized(failing).withOutputDataset(notEmpty);
        assertThrows(IllegalArgumentException.class, serializedBuilder::build);
      }
      assertFalse(failing.closed);
    }
  }

  @Test
  public void fromDatasetRejectsAnArrayItCannotCopy() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create()) {
      var empty = CagraIndex.newBuilder(resources).fromDataset(new float[0][]);
      assertThrows(IllegalArgumentException.class, empty::build);
      var ragged = CagraIndex.newBuilder(resources).fromDataset(new float[][] {{1, 2}, {3, 4, 5}});
      var failure = assertThrows(IllegalArgumentException.class, ragged::build);
      assertTrue(failure.getMessage(), failure.getMessage().contains("vectors[1] has 3"));
    }
  }

  @Test
  public void requiredInputsMustNotBeNull() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create()) {
      var matrix = fake(CuVSMatrix.class);
      var quantizer = quantizer(BbqQuantizer.CodeLayout.PACKED_1B);
      var builder = CagraIndex.newBuilder(resources);
      assertThrows(NullPointerException.class, () -> builder.fromDataset((CuVSMatrix) null));
      assertThrows(NullPointerException.class, () -> builder.fromDataset((float[][]) null));
      assertThrows(NullPointerException.class, () -> builder.fromBbq(null));
      assertThrows(NullPointerException.class, () -> builder.fromBbq(quantizer, null));
      assertThrows(NullPointerException.class, () -> builder.fromGraph(null, matrix, matrix));
      assertThrows(
          NullPointerException.class,
          () -> builder.fromGraph(CuvsDistanceType.L2Expanded, null, matrix));
      assertThrows(NullPointerException.class, () -> builder.fromSerialized(null));
    }
  }

  /** A quantizer whose matrices are never read, since the checks here fail before that. */
  static BbqQuantizer quantizer(BbqQuantizer.CodeLayout layout) {
    CuVSMatrix component = fake(CuVSMatrix.class);
    return new BbqQuantizer.Builder()
        .withCodes(component)
        .withLowerIntervals(component)
        .withUpperIntervals(component)
        .withAdditionalCorrections(component)
        .withQuantizedComponentSums(component)
        .withCentroid(component)
        .withDequantDelta(component)
        .withDequantSumDelta(component)
        .withRowNorm(component)
        .withLayout(layout)
        .withMetric(CuvsDistanceType.L2Expanded)
        .withCentroidNormSq(0.0f)
        .build();
  }

  /** A stream that records whether it was closed. */
  static final class CloseTrackingStream extends ByteArrayInputStream {
    boolean closed;

    CloseTrackingStream(byte[] bytes) {
      super(bytes);
    }

    @Override
    public void close() {
      closed = true;
    }
  }

  /** An instance that only answers the methods of {@link Object}. */
  static <T> T fake(Class<T> type) {
    return type.cast(
        Proxy.newProxyInstance(
            type.getClassLoader(),
            new Class<?>[] {type},
            (proxy, method, args) ->
                switch (method.getName()) {
                  case "toString" -> "fake " + type.getSimpleName();
                  case "hashCode" -> System.identityHashCode(proxy);
                  case "equals" -> proxy == args[0];
                  default -> null;
                }));
  }
}

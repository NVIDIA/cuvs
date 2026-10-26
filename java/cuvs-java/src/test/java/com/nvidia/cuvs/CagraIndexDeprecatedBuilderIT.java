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
import java.io.InputStream;
import org.junit.Before;
import org.junit.Test;
import org.junit.function.ThrowingRunnable;
import org.junit.runner.RunWith;

/**
 * The deprecated setters of {@link CagraIndex.Builder}, end to end through the real provider, for
 * as long as they are supported. Delete this class together with them.
 */
@RunWith(RandomizedRunner.class)
@SuppressWarnings("removal")
public class CagraIndexDeprecatedBuilderIT extends CuVSTestCase {

  private static final int ROWS = 512;
  private static final int DIM = 16;

  private float[][] vectors;

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxSupportedArch());
    initializeRandom();
    vectors = generateData(random, ROWS, DIM);
  }

  @Test
  public void buildsFromADataset() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create();
        CagraIndex index =
            CagraIndex.newBuilder(resources)
                .withDataset(vectors)
                .withIndexParams(params())
                .build()) {
      assertEquals(ROWS, index.size());
    }
  }

  @Test
  public void loadsASerializedIndex() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create()) {
      CuVSMatrix dataset = CuVSMatrix.ofArray(vectors);
      // The file keeps the layout the index was built with, padded only if the rows already were.
      boolean padded = CagraIndex.isPaddedDataset(dataset);
      byte[] serialized;
      try (CagraIndex index =
          CagraIndex.newBuilder(resources).fromDataset(dataset).withIndexParams(params()).build()) {
        var bytes = new ByteArrayOutputStream();
        index.serialize(bytes);
        serialized = bytes.toByteArray();
      }

      // Unlike fromSerialized, the deprecated from(InputStream) closes the stream, as it always
      // has.
      var stream = new CagraIndexBuilderIT.CloseTrackingStream(serialized);
      try (CagraIndex loaded = CagraIndex.newBuilder(resources).from(stream).build()) {
        assertEquals(ROWS, loaded.size());
      }
      assertTrue(stream.closed);
      try (CagraIndex.DeserializeDataset outDataset =
              padded ? new CagraIndex.PaddedDataset() : new CagraIndex.StandardDataset();
          CagraIndex loaded =
              CagraIndex.newBuilder(resources)
                  .from(new ByteArrayInputStream(serialized), outDataset)
                  .build()) {
        assertEquals(ROWS, loaded.size());
        assertTrue("the caller should own the loaded dataset", outDataset.isPresent());
      }
      // As before, a later from(InputStream) drops the output dataset an earlier call gave.
      try (var dropped = new CagraIndex.StandardDataset();
          CagraIndex loaded =
              CagraIndex.newBuilder(resources)
                  .from(new ByteArrayInputStream(serialized), dropped)
                  .from(new ByteArrayInputStream(serialized))
                  .build()) {
        assertFalse(dropped.isPresent());
      }
    }
  }

  @Test
  public void createsAnIndexFromAGraph() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create();
        CagraIndex index =
            CagraIndex.newBuilder(resources)
                .fromDataset(vectors)
                .withIndexParams(params())
                .build();
        // A host copy, so the new index doesn't depend on the first one's device graph.
        CuVSMatrix graph = index.getGraph().toHost()) {
      // Owned by the new index once built, so it is not closed here.
      CuVSMatrix deviceVectors;
      try (CuVSMatrix hostVectors = CuVSMatrix.ofArray(vectors)) {
        deviceVectors = hostVectors.toDevice(resources);
      }
      try (CagraIndex fromGraph =
          CagraIndex.newBuilder(resources)
              .from(graph)
              .withDataset(deviceVectors)
              .withIndexParams(params())
              .build()) {
        assertEquals(ROWS, fromGraph.size());
        assertEquals(index.getGraphDegree(), fromGraph.getGraphDegree());
      }
    }
  }

  /** Without withIndexParams this used to fall back on native defaults that could not build. */
  @Test
  public void buildsWithoutParameters() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create();
        CagraIndex index = CagraIndex.newBuilder(resources).withDataset(vectors).build()) {
      assertEquals(ROWS, index.size());
    }
  }

  @Test
  public void keepsTheLastCall() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create();
        CagraIndex index =
            CagraIndex.newBuilder(resources)
                .withIndexParams(new CagraIndexParams.Builder().withGraphDegree(8).build())
                .withDataset(generateData(random, ROWS / 2, DIM))
                .withDataset(CuVSMatrix.ofArray(vectors))
                .withIndexParams(params())
                .build()) {
      assertEquals(ROWS, index.size());
      assertEquals(16, index.getGraphDegree());
    }
  }

  /** A failed build leaves the inputs in place, so the builder can be fixed and retried. */
  @Test
  public void canRetryAFailedBuild() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create()) {
      var builder =
          CagraIndex.newBuilder(resources)
              .withDataset(vectors)
              .withIndexParams(params())
              .from(new ByteArrayInputStream(new byte[0]));
      // Loading the stream would ignore the dataset.
      assertThrows(IllegalArgumentException.class, builder::build);
      builder.from((InputStream) null);
      try (CagraIndex index = builder.build()) {
        assertEquals(ROWS, index.size());
      }
    }
  }

  /** The array is copied when it is given, so a bad one fails right there. */
  @Test
  public void copiesAnArrayRightAway() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create()) {
      var builder = CagraIndex.newBuilder(resources);
      assertThrows(IllegalArgumentException.class, () -> builder.withDataset(new float[0][]));
    }
  }

  @Test
  public void rejectsEveryInputALoadWouldIgnore() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create()) {
      var matrix = CagraIndexBuilderIT.fake(CuVSMatrix.class);
      var empty = new ByteArrayInputStream(new byte[0]);
      assertIgnoredWhenLoading(
          () -> CagraIndex.newBuilder(resources).from(empty).withDataset(matrix).build());
      assertIgnoredWhenLoading(
          () -> CagraIndex.newBuilder(resources).withDataset(vectors).from(empty).build());
      assertIgnoredWhenLoading(
          () -> CagraIndex.newBuilder(resources).from(empty).from(matrix).build());
      assertIgnoredWhenLoading(
          () ->
              CagraIndex.newBuilder(resources)
                  .from(empty)
                  .withBbqDataset(CagraIndexBuilderIT.quantizer(BbqQuantizer.CodeLayout.PACKED_1B))
                  .build());
      assertIgnoredWhenLoading(
          () -> CagraIndex.newBuilder(resources).from(empty).withIndexParams(params()).build());
    }
  }

  /** from(InputStream) closes the stream, as documented, even when an input is rejected. */
  @Test
  public void closesTheStreamWhenALoadIsRejected() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create()) {
      var stream = new CagraIndexBuilderIT.CloseTrackingStream(new byte[0]);
      assertIgnoredWhenLoading(
          () -> CagraIndex.newBuilder(resources).from(stream).withIndexParams(params()).build());
      assertTrue(stream.closed);
    }
  }

  @Test
  public void rejectsQuantizersWhenCreatingAnIndexFromAGraph() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create()) {
      var matrix = CagraIndexBuilderIT.fake(CuVSMatrix.class);
      var failure =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  CagraIndex.newBuilder(resources)
                      .from(matrix)
                      .withDataset(matrix)
                      .withIndexParams(params())
                      .withBbqDataset(
                          CagraIndexBuilderIT.quantizer(BbqQuantizer.CodeLayout.PACKED_1B))
                      .build());
      assertTrue(failure.getMessage(), failure.getMessage().contains("fromGraph"));
    }
  }

  @Test
  public void cannotBeMixedWithFrom() throws Throwable {
    try (CuVSResources resources = CheckedCuVSResources.create()) {
      var afterSetter = CagraIndex.newBuilder(resources).withIndexParams(params());
      var failure =
          assertThrows(IllegalStateException.class, () -> afterSetter.fromDataset(vectors));
      assertTrue(failure.getMessage(), failure.getMessage().contains("withIndexParams"));

      var afterFrom = CagraIndex.newBuilder(resources);
      afterFrom.fromDataset(vectors);
      failure = assertThrows(IllegalStateException.class, () -> afterFrom.withDataset(vectors));
      assertTrue(failure.getMessage(), failure.getMessage().contains("fromDataset"));
    }
  }

  private static void assertIgnoredWhenLoading(ThrowingRunnable build) {
    var failure = assertThrows(IllegalArgumentException.class, build);
    assertTrue(failure.getMessage(), failure.getMessage().contains("fromSerialized"));
  }

  private static CagraIndexParams params() {
    return new CagraIndexParams.Builder()
        .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
        .withGraphDegree(16)
        .withIntermediateGraphDegree(32)
        .withMetric(CuvsDistanceType.L2Expanded)
        .build();
  }
}

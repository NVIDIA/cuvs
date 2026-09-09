/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;
import static org.junit.Assert.*;

import com.carrotsearch.randomizedtesting.RandomizedRunner;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.Random;
import java.util.stream.Stream;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/**
 * Build and serialize tests for {@link VamanaIndex}.
 *
 * <p>cuVS exposes no Vamana search entry point, so these tests cover
 * construction, dimensions, the DiskANN file layout produced by serialization,
 * parameter validation, and lifecycle.
 */
@RunWith(RandomizedRunner.class)
public class VamanaBuildAndSerializeIT extends CuVSTestCase {

  private static final int ROWS = 1000;
  private static final int DIMENSIONS = 32;
  private static final int GRAPH_DEGREE = 32;

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxAmd64());
    initializeRandom();
  }

  private static float[][] randomFloatDataset() {
    Random random = new Random(42);
    float[][] dataset = new float[ROWS][DIMENSIONS];
    for (int i = 0; i < ROWS; i++) {
      for (int j = 0; j < DIMENSIONS; j++) {
        dataset[i][j] = random.nextFloat();
      }
    }
    return dataset;
  }

  private static VamanaIndexParams defaultParams() {
    return new VamanaIndexParams.Builder()
        .withGraphDegree(GRAPH_DEGREE)
        .withVisitedSize(64)
        .build();
  }

  /** Runs the body against a fresh output prefix and removes every file afterwards. */
  private static void withPrefix(PrefixConsumer body) throws Throwable {
    Path dir = Files.createTempDirectory("cuvs-vamana");
    try {
      body.accept(dir.resolve("index"));
    } finally {
      deleteRecursively(dir);
    }
  }

  private interface PrefixConsumer {
    void accept(Path prefix) throws Throwable;
  }

  private static void deleteRecursively(Path root) throws IOException {
    if (!Files.exists(root)) {
      return;
    }
    try (Stream<Path> paths = Files.walk(root)) {
      paths
          .sorted(Comparator.reverseOrder())
          .forEach(
              path -> {
                try {
                  Files.deleteIfExists(path);
                } catch (IOException e) {
                  throw new UncheckedIOExceptionWrapper(e);
                }
              });
    } catch (UncheckedIOExceptionWrapper e) {
      throw e.cause;
    }
  }

  private static final class UncheckedIOExceptionWrapper extends RuntimeException {
    private final IOException cause;

    UncheckedIOExceptionWrapper(IOException cause) {
      super(cause);
      this.cause = cause;
    }
  }

  private static long sizeOf(Path path) throws IOException {
    assertTrue(path + " should exist", Files.exists(path));
    long size = Files.size(path);
    assertTrue(path + " should not be empty", size > 0);
    return size;
  }

  private static Path dataFile(Path prefix) {
    return prefix.resolveSibling(prefix.getFileName() + ".data");
  }

  private static ByteBuffer readHead(Path path, int bytes) throws IOException {
    byte[] head = new byte[bytes];
    try (var in = Files.newInputStream(path)) {
      assertEquals(bytes, in.readNBytes(head, 0, bytes));
    }
    return ByteBuffer.wrap(head).order(ByteOrder.LITTLE_ENDIAN);
  }

  private static CuVSMatrix hostMatrix(CuVSMatrix.DataType dataType) {
    return fill(CuVSMatrix.hostBuilder(ROWS, DIMENSIONS, dataType), dataType);
  }

  private static CuVSMatrix deviceMatrix(CuVSResources resources, CuVSMatrix.DataType dataType) {
    return fill(CuVSMatrix.deviceBuilder(resources, ROWS, DIMENSIONS, dataType), dataType);
  }

  private static CuVSMatrix fill(
      CuVSMatrix.Builder<? extends CuVSMatrix> builder, CuVSMatrix.DataType dataType) {
    Random random = new Random(7);
    for (int i = 0; i < ROWS; i++) {
      switch (dataType) {
        case FLOAT -> {
          float[] row = new float[DIMENSIONS];
          for (int j = 0; j < DIMENSIONS; j++) {
            row[j] = random.nextFloat();
          }
          builder.addVector(row);
        }
        case BYTE -> {
          byte[] row = new byte[DIMENSIONS];
          random.nextBytes(row);
          builder.addVector(row);
        }
        case HALF -> {
          short[] row = new short[DIMENSIONS];
          for (int j = 0; j < DIMENSIONS; j++) {
            row[j] = Float.floatToFloat16(random.nextFloat());
          }
          builder.addVector(row);
        }
        case INT -> {
          int[] row = new int[DIMENSIONS];
          for (int j = 0; j < DIMENSIONS; j++) {
            row[j] = random.nextInt(100);
          }
          builder.addVector(row);
        }
        default -> throw new IllegalArgumentException("unhandled type " + dataType);
      }
    }
    return builder.build();
  }

  @Test
  public void testBuildFloatAndGetDimensions() throws Throwable {
    try (CuVSResources resources = CuVSResources.create();
        VamanaIndex index =
            VamanaIndex.newBuilder(resources)
                .withDataset(randomFloatDataset())
                .withIndexParams(defaultParams())
                .build()) {
      assertEquals(DIMENSIONS, index.getDimensions());
    }
  }

  @Test
  public void testBuildUnsignedByteDataset() throws Throwable {
    try (CuVSResources resources = CuVSResources.create();
        CuVSMatrix dataset = hostMatrix(CuVSMatrix.DataType.BYTE);
        VamanaIndex index =
            VamanaIndex.newBuilder(resources)
                .withDataset(dataset)
                .withIndexParams(defaultParams())
                .build()) {
      assertEquals(DIMENSIONS, index.getDimensions());

      withPrefix(
          prefix -> {
            index.serialize(prefix, true);
            // one byte per component, plus the two 32-bit header values
            assertEquals((long) ROWS * DIMENSIONS + 8, sizeOf(dataFile(prefix)));
          });
    }
  }

  @Test
  public void testSerializeWritesGraphAndDataset() throws Throwable {
    try (CuVSResources resources = CuVSResources.create();
        VamanaIndex index =
            VamanaIndex.newBuilder(resources)
                .withDataset(randomFloatDataset())
                .withIndexParams(defaultParams())
                .build()) {

      withPrefix(
          prefix -> {
            index.serialize(prefix, true);

            long graphSize = sizeOf(prefix);
            Path data = dataFile(prefix);

            // the DiskANN .data file is two 32-bit header values followed by the
            // raw vectors
            ByteBuffer dataHead = readHead(data, 8);
            assertEquals(ROWS, dataHead.getInt());
            assertEquals(DIMENSIONS, dataHead.getInt());
            assertEquals((long) ROWS * DIMENSIONS * Float.BYTES + 8, sizeOf(data));

            // the graph file opens with its own length, then the observed
            // maximum degree, which the configured graph degree bounds
            ByteBuffer graphHead = readHead(prefix, 12);
            assertEquals(graphSize, graphHead.getLong());
            int maxDegree = graphHead.getInt();
            assertTrue("max degree should be positive, was " + maxDegree, maxDegree > 0);
            assertTrue(
                "max degree " + maxDegree + " should not exceed the configured graph degree",
                maxDegree <= GRAPH_DEGREE);
          });
    }
  }

  @Test
  public void testSerializeWithoutDataset() throws Throwable {
    try (CuVSResources resources = CuVSResources.create();
        VamanaIndex index =
            VamanaIndex.newBuilder(resources)
                .withDataset(randomFloatDataset())
                .withIndexParams(defaultParams())
                .build()) {

      withPrefix(
          prefix -> {
            index.serialize(prefix, false);
            sizeOf(prefix);
            assertFalse(
                "the dataset file should not be written when includeDataset is false",
                Files.exists(dataFile(prefix)));
          });
    }
  }

  @Test
  public void testDefaultParametersAreUsedWhenNoneAreGiven() throws Throwable {
    try (CuVSResources resources = CuVSResources.create();
        VamanaIndex index =
            VamanaIndex.newBuilder(resources).withDataset(randomFloatDataset()).build()) {
      assertEquals(DIMENSIONS, index.getDimensions());
    }
  }

  @Test
  public void testUnsupportedGraphDegreeIsRejectedInJava() {
    var builder = new VamanaIndexParams.Builder().withGraphDegree(48).withVisitedSize(128);
    IllegalArgumentException e = assertThrows(IllegalArgumentException.class, builder::build);
    assertTrue(e.getMessage(), e.getMessage().contains("graphDegree"));
  }

  @Test
  public void testVisitedSizeBelowGraphDegreeIsRejectedInJava() {
    var builder = new VamanaIndexParams.Builder().withGraphDegree(64).withVisitedSize(8);
    IllegalArgumentException e = assertThrows(IllegalArgumentException.class, builder::build);
    assertTrue(e.getMessage(), e.getMessage().contains("visitedSize"));
  }

  @Test
  public void testInvalidVamanaItersIsRejectedInJava() {
    var builder = new VamanaIndexParams.Builder().withVamanaIters(0.5f);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void testInvalidQueueSizeIsRejectedInJava() {
    var builder = new VamanaIndexParams.Builder().withQueueSize(100);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void testSupportedGraphDegreesAreDefensivelyCopied() {
    int[] degrees = VamanaIndexParams.supportedGraphDegrees();
    assertArrayEquals(new int[] {32, 64, 128, 256}, degrees);
    degrees[0] = -1;
    assertArrayEquals(new int[] {32, 64, 128, 256}, VamanaIndexParams.supportedGraphDegrees());
  }

  @Test
  public void testUnsupportedDataTypeIsRejected() throws Throwable {
    try (CuVSResources resources = CuVSResources.create();
        CuVSMatrix dataset = hostMatrix(CuVSMatrix.DataType.INT)) {
      var builder =
          VamanaIndex.newBuilder(resources).withDataset(dataset).withIndexParams(defaultParams());
      IllegalArgumentException e = assertThrows(IllegalArgumentException.class, builder::build);
      assertTrue(e.getMessage(), e.getMessage().contains("FLOAT, HALF, and BYTE"));
    }
  }

  @Test
  public void testMissingDatasetIsRejected() throws Throwable {
    try (CuVSResources resources = CuVSResources.create()) {
      var builder = VamanaIndex.newBuilder(resources).withIndexParams(defaultParams());
      assertThrows(IllegalArgumentException.class, builder::build);
    }
  }

  @Test
  public void testCallerSuppliedDatasetOutlivesTheIndex() throws Throwable {
    // the native index may retain a non-owning device view, so a caller
    // supplied matrix is the caller's to close
    try (CuVSResources resources = CuVSResources.create();
        CuVSMatrix dataset = hostMatrix(CuVSMatrix.DataType.FLOAT)) {
      try (VamanaIndex index =
          VamanaIndex.newBuilder(resources)
              .withDataset(dataset)
              .withIndexParams(defaultParams())
              .build()) {
        assertEquals(DIMENSIONS, index.getDimensions());
      }
      // still usable after the index is closed
      assertEquals(ROWS, dataset.size());
    }
  }

  @Test
  public void testBuilderReuseDoesNotStrandAnOwnedDataset() throws Throwable {
    try (CuVSResources resources = CuVSResources.create()) {
      var builder = VamanaIndex.newBuilder(resources).withIndexParams(defaultParams());
      // the first matrix is created and then replaced, which must release it
      builder.withDataset(randomFloatDataset());
      builder.withDataset(randomFloatDataset());
      try (VamanaIndex index = builder.build()) {
        assertEquals(DIMENSIONS, index.getDimensions());
      }
      // the builder handed ownership to the index, so it now has no dataset
      assertThrows(IllegalArgumentException.class, builder::build);
    }
  }

  @Test
  public void testBuildHalfDatasetOnDevice() throws Throwable {
    // covers float16 and the device-backed matrix path together, since the
    // native index may retain a non-owning device view of this matrix
    try (CuVSResources resources = CuVSResources.create();
        CuVSMatrix dataset = deviceMatrix(resources, CuVSMatrix.DataType.HALF);
        VamanaIndex index =
            VamanaIndex.newBuilder(resources)
                .withDataset(dataset)
                .withIndexParams(defaultParams())
                .build()) {
      assertEquals(DIMENSIONS, index.getDimensions());

      withPrefix(
          prefix -> {
            index.serialize(prefix, true);
            // two bytes per component, plus the two 32-bit header values
            assertEquals((long) ROWS * DIMENSIONS * 2 + 8, sizeOf(dataFile(prefix)));
          });
    }
  }

  @Test
  public void testBuildFloatDatasetOnDevice() throws Throwable {
    try (CuVSResources resources = CuVSResources.create();
        CuVSMatrix dataset = deviceMatrix(resources, CuVSMatrix.DataType.FLOAT)) {
      try (VamanaIndex index =
          VamanaIndex.newBuilder(resources)
              .withDataset(dataset)
              .withIndexParams(defaultParams())
              .build()) {
        assertEquals(DIMENSIONS, index.getDimensions());
      }
      // the caller's device matrix outlives the index
      assertEquals(ROWS, dataset.size());
    }
  }

  @Test
  public void testSqrtL2MetricIsAccepted() throws Throwable {
    VamanaIndexParams params =
        new VamanaIndexParams.Builder()
            .withGraphDegree(GRAPH_DEGREE)
            .withVisitedSize(64)
            .withMetric(VamanaIndexParams.CuvsDistanceType.L2SqrtExpanded)
            .build();
    try (CuVSResources resources = CuVSResources.create();
        VamanaIndex index =
            VamanaIndex.newBuilder(resources)
                .withDataset(randomFloatDataset())
                .withIndexParams(params)
                .build()) {
      assertEquals(DIMENSIONS, index.getDimensions());
    }
  }

  @Test
  public void testInfiniteVamanaItersIsRejectedInJava() {
    var builder = new VamanaIndexParams.Builder().withVamanaIters(Float.POSITIVE_INFINITY);
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void testUseAfterCloseIsRejected() throws Throwable {
    try (CuVSResources resources = CuVSResources.create()) {
      VamanaIndex index =
          VamanaIndex.newBuilder(resources)
              .withDataset(randomFloatDataset())
              .withIndexParams(defaultParams())
              .build();
      index.close();
      assertThrows(IllegalStateException.class, index::getDimensions);
      assertThrows(IllegalStateException.class, index::close);
    }
  }
}

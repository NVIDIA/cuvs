/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;
import static org.junit.Assert.*;

import com.nvidia.cuvs.spi.CuVSProvider;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;
import org.junit.Before;
import org.junit.Test;

public class CuVSMatrixFailureCleanupIT extends CuVSTestCase {

  // JDKProvider uses a buffered builder when one row fits in its pinned buffer;
  // larger rows use a direct-copy builder. Both have contiguous and strided constructors.
  private static final int PINNED_BUFFER_BYTES = 8 * 1024 * 1024;
  private static final int BUFFERED_ROW_COLUMNS = 4;
  private static final int DIRECT_COPY_ROW_COLUMNS = PINNED_BUFFER_BYTES / Float.BYTES + 1;
  private static final int DEFAULT_COLUMN_STRIDE = -1;

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxSupportedArch());
  }

  @Test
  public void testFailedHostToDeviceCopyReleasesDestination() throws Throwable {
    RuntimeException copyFailure = new IllegalStateException("injected device-copy failure");
    AtomicReference<CuVSMatrix> allocatedDestination = new AtomicReference<>();
    CuVSHostMatrix fakeHostSource =
        fakeFailingCopySource(CuVSHostMatrix.class, allocatedDestination, copyFailure);
    assertWorkspaceReleased(
        resources -> {
          var thrown =
              assertThrows(IllegalStateException.class, () -> fakeHostSource.toDevice(resources));
          assertSame("conversion must preserve the copy failure", copyFailure, thrown);
          assertNotNull("copy did not reach an allocated destination", allocatedDestination.get());
        });
  }

  @Test
  public void testFailedDeviceToHostCopyClosesDestinationArena() {
    RuntimeException copyFailure = new IllegalStateException("injected host-copy failure");
    AtomicReference<CuVSMatrix> allocatedDestination = new AtomicReference<>();
    CuVSDeviceMatrix fakeDeviceSource =
        fakeFailingCopySource(CuVSDeviceMatrix.class, allocatedDestination, copyFailure);

    var thrown = assertThrows(IllegalStateException.class, fakeDeviceSource::toHost);
    assertSame("conversion must preserve the copy failure", copyFailure, thrown);
    assertNotNull("copy did not reach an allocated destination", allocatedDestination.get());
    // A closed host arena rejects reads. The outer close also prevents a test leak if this fails.
    try (CuVSMatrix hostDestination = allocatedDestination.get()) {
      assertThrows(
          "the failed conversion retained its host arena",
          IllegalStateException.class,
          () -> hostDestination.getRow(0).getAsFloat(0));
    }
  }

  @Test
  public void testBufferedBuilderReleasesMatrixWhenStreamLookupFails() throws Throwable {
    assertFailedBuilderReleasesMatrix(
        resources ->
            CuVSMatrix.deviceBuilder(
                resources, 1, BUFFERED_ROW_COLUMNS, CuVSMatrix.DataType.FLOAT));
  }

  @Test
  public void testStridedBufferedBuilderReleasesMatrixWhenStreamLookupFails() throws Throwable {
    int rowStrideWithOnePaddingElement = BUFFERED_ROW_COLUMNS + 1;
    assertFailedBuilderReleasesMatrix(
        resources ->
            CuVSMatrix.deviceBuilder(
                resources,
                1,
                BUFFERED_ROW_COLUMNS,
                rowStrideWithOnePaddingElement,
                DEFAULT_COLUMN_STRIDE,
                CuVSMatrix.DataType.FLOAT));
  }

  @Test
  public void testDirectCopyBuilderReleasesMatrixWhenStreamLookupFails() throws Throwable {
    assertFailedBuilderReleasesMatrix(
        resources ->
            CuVSMatrix.deviceBuilder(
                resources, 1, DIRECT_COPY_ROW_COLUMNS, CuVSMatrix.DataType.FLOAT));
  }

  @Test
  public void testStridedDirectCopyBuilderReleasesMatrixWhenStreamLookupFails() throws Throwable {
    int rowStrideWithOnePaddingElement = DIRECT_COPY_ROW_COLUMNS + 1;
    assertFailedBuilderReleasesMatrix(
        resources ->
            CuVSMatrix.deviceBuilder(
                resources,
                1,
                DIRECT_COPY_ROW_COLUMNS,
                rowStrideWithOnePaddingElement,
                DEFAULT_COLUMN_STRIDE,
                CuVSMatrix.DataType.FLOAT));
  }

  @Test
  public void testArrayFactoriesRejectRowsShorterThanTheFirstRow() {
    // Cleanup must not change the error reported by the existing bulk-copy implementation.
    assertThrows(
        IndexOutOfBoundsException.class, () -> CuVSMatrix.ofArray(new float[][] {{1, 2}, {3}}));
    assertThrows(
        IndexOutOfBoundsException.class, () -> CuVSMatrix.ofArray(new int[][] {{1, 2}, {3}}));
    assertThrows(
        IndexOutOfBoundsException.class, () -> CuVSMatrix.ofArray(new byte[][] {{1, 2}, {3}}));
  }

  @Test
  public void testArrayFactoriesIgnoreExtraColumnsInLaterRows() {
    // ofArray historically copies only the first row's width, unlike builder.addVector.
    try (CuVSMatrix matrix = CuVSMatrix.ofArray(new float[][] {{1, 2}, {3, 4, 5}})) {
      float[][] copied = new float[2][2];
      matrix.toArray(copied);
      assertArrayEquals(new float[] {3, 4}, copied[1], 0f);
    }
    try (CuVSMatrix matrix = CuVSMatrix.ofArray(new int[][] {{1, 2}, {3, 4, 5}})) {
      int[][] copied = new int[2][2];
      matrix.toArray(copied);
      assertArrayEquals(new int[] {3, 4}, copied[1]);
    }
    try (CuVSMatrix matrix = CuVSMatrix.ofArray(new byte[][] {{1, 2}, {3, 4, 5}})) {
      byte[][] copied = new byte[2][2];
      matrix.toArray(copied);
      assertArrayEquals(new byte[] {3, 4}, copied[1]);
    }
  }

  @Test
  public void testHostBuilderRejectsRowStrideShorterThanTheRow() {
    int columns = 4;
    int rowStrideShorterThanRow = columns - 1;
    assertThrows(
        IllegalArgumentException.class,
        () ->
            CuVSMatrix.hostBuilder(
                2,
                columns,
                rowStrideShorterThanRow,
                DEFAULT_COLUMN_STRIDE,
                CuVSMatrix.DataType.FLOAT));
  }

  @Test
  public void testHostBuilderRejectsUnsupportedColumnStride() {
    int defaultRowStride = -1;
    int unsupportedColumnStride = 2;
    assertThrows(
        UnsupportedOperationException.class,
        () ->
            CuVSMatrix.hostBuilder(
                2, 4, defaultRowStride, unsupportedColumnStride, CuVSMatrix.DataType.FLOAT));
  }

  private static void assertFailedBuilderReleasesMatrix(
      Function<CuVSResources, CuVSMatrix.Builder<CuVSDeviceMatrix>> createBuilder)
      throws Throwable {
    RuntimeException streamFailure = new IllegalStateException("injected stream-lookup failure");
    assertWorkspaceReleased(
        resources -> {
          CuVSResources failingResources = resourcesFailingStreamLookup(resources, streamFailure);
          var thrown =
              assertThrows(
                  IllegalStateException.class, () -> createBuilder.apply(failingResources));
          assertSame("constructor must preserve the stream failure", streamFailure, thrown);
        });
  }

  /** Allow real allocation, then fail stream lookup during builder initialization. */
  private static CuVSResources resourcesFailingStreamLookup(
      CuVSResources realResources, RuntimeException streamFailure) {
    return (CuVSResources)
        Proxy.newProxyInstance(
            CuVSResources.class.getClassLoader(),
            new Class<?>[] {CuVSResources.class},
            (proxy, method, arguments) -> {
              if (method.getName().equals("access") && isLookingUpCudaStream()) {
                throw streamFailure;
              }
              try {
                return method.invoke(realResources, arguments);
              } catch (InvocationTargetException failure) {
                throw failure.getCause();
              }
            });
  }

  private static boolean isLookingUpCudaStream() {
    // access() is also used for allocation; only Util.getStream must fail.
    return StackWalker.getInstance()
        .walk(
            frames ->
                frames.anyMatch(
                    frame ->
                        frame.getClassName().equals("com.nvidia.cuvs.internal.common.Util")
                            && frame.getMethodName().equals("getStream")));
  }

  /**
   * Fake source metadata and copying, but real default conversion and destination allocation.
   * invokeDefault runs toHost()/toDevice(resources), which allocates the destination before
   * calling the copy overload intercepted below. Capturing that destination proves copying was reached.
   */
  private static <T extends CuVSMatrix> T fakeFailingCopySource(
      Class<T> type, AtomicReference<CuVSMatrix> destination, RuntimeException failure) {
    return type.cast(
        Proxy.newProxyInstance(
            type.getClassLoader(),
            new Class<?>[] {type},
            (proxy, method, arguments) -> {
              if (method.isDefault()) {
                return InvocationHandler.invokeDefault(proxy, method, arguments);
              }
              return switch (method.getName()) {
                case "size" -> 1L;
                case "columns" -> 4L;
                case "dataType" -> CuVSMatrix.DataType.FLOAT;
                case "toHost", "toDevice" -> {
                  destination.set((CuVSMatrix) arguments[0]);
                  throw failure;
                }
                default -> throw new AssertionError("Unexpected fake-source operation: " + method);
              };
            }));
  }

  @FunctionalInterface
  private interface TrackedOperation {
    void run(CuVSResources resources) throws Throwable;
  }

  private static void assertWorkspaceReleased(TrackedOperation operation) throws Throwable {
    Path csv = Files.createTempFile("matrix-failure-memory", ".csv");
    try {
      try (CuVSResources resources =
          CuVSResources.create(CuVSProvider.tempDirectory(), csv, Duration.ofMillis(1))) {
        operation.run(resources);
      }
      List<String> lines = Files.readAllLines(csv);
      assertTrue("memory tracker did not write a sample", lines.size() >= 2);
      List<String> header = List.of(lines.getFirst().split(",", -1));
      String[] lastSample = lines.getLast().split(",", -1);
      long allocatedBytes = csvLong(header, lastSample, "workspace_total_alloc");
      long remainingBytes = csvLong(header, lastSample, "workspace_current");
      long freedBytes = csvLong(header, lastSample, "workspace_total_freed");
      assertTrue("the test did not allocate device memory", allocatedBytes > 0);
      assertEquals("failed operation retained device memory", 0, remainingBytes);
      assertEquals("every allocated byte must be freed", allocatedBytes, freedBytes);
    } finally {
      Files.deleteIfExists(csv);
    }
  }

  private static long csvLong(List<String> header, String[] sample, String column) {
    int index = header.indexOf(column);
    assertTrue("missing memory-tracker column: " + column, index >= 0);
    return Long.parseLong(sample[index]);
  }
}

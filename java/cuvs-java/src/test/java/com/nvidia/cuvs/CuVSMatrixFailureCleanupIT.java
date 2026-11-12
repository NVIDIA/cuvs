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
import org.junit.Before;
import org.junit.Test;

public class CuVSMatrixFailureCleanupIT extends CuVSTestCase {

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxSupportedArch());
  }

  @Test
  public void testFailedHostToDeviceCopyReleasesDestination() throws Throwable {
    RuntimeException copyFailure = new IllegalStateException("injected device-copy failure");
    AtomicReference<CuVSMatrix> destination = new AtomicReference<>();
    CuVSHostMatrix source = fakeFailingCopySource(CuVSHostMatrix.class, destination, copyFailure);
    assertWorkspaceReleased(
        resources -> {
          assertSame(
              copyFailure,
              assertThrows(IllegalStateException.class, () -> source.toDevice(resources)));
          assertNotNull("copy did not reach an allocated destination", destination.get());
        });
  }

  @Test
  public void testFailedDeviceToHostCopyClosesDestinationArena() {
    RuntimeException copyFailure = new IllegalStateException("injected host-copy failure");
    AtomicReference<CuVSMatrix> destination = new AtomicReference<>();
    CuVSDeviceMatrix source =
        fakeFailingCopySource(CuVSDeviceMatrix.class, destination, copyFailure);

    assertSame(copyFailure, assertThrows(IllegalStateException.class, source::toHost));
    assertNotNull("copy did not reach an allocated destination", destination.get());
    try (CuVSMatrix allocatedDestination = destination.get()) {
      assertThrows(
          "the failed conversion retained its host arena",
          IllegalStateException.class,
          () -> allocatedDestination.getRow(0).getAsFloat(0));
    }
  }

  @Test
  public void testFailedDeviceBuilderInitializationReleasesMatrix() throws Throwable {
    // JDKProvider uses direct copies for rows larger than its 8 MiB pinned buffer.
    int bufferedRowByteLimit = 8 * 1024 * 1024;
    int directCopyColumns = bufferedRowByteLimit / Float.BYTES + 1;
    for (int columns : new int[] {4, directCopyColumns}) {
      assertFailedBuilderReleasesMatrix(columns, false);
      assertFailedBuilderReleasesMatrix(columns, true);
    }
  }

  private static void assertFailedBuilderReleasesMatrix(int columns, boolean strided)
      throws Throwable {
    RuntimeException streamFailure = new IllegalStateException("injected stream-lookup failure");
    assertWorkspaceReleased(
        resources -> {
          CuVSResources failingResources =
              (CuVSResources)
                  Proxy.newProxyInstance(
                      CuVSResources.class.getClassLoader(),
                      new Class<?>[] {CuVSResources.class},
                      (proxy, method, arguments) -> {
                        // Allocation succeeds first; fail only in the subsequent stream lookup.
                        boolean lookingUpStream =
                            StackWalker.getInstance()
                                .walk(
                                    frames ->
                                        frames.anyMatch(
                                            frame ->
                                                frame
                                                        .getClassName()
                                                        .equals(
                                                            "com.nvidia.cuvs.internal.common.Util")
                                                    && frame.getMethodName().equals("getStream")));
                        if (method.getName().equals("access") && lookingUpStream) {
                          throw streamFailure;
                        }
                        try {
                          return method.invoke(resources, arguments);
                        } catch (InvocationTargetException failure) {
                          throw failure.getCause();
                        }
                      });
          assertSame(
              streamFailure,
              assertThrows(
                  IllegalStateException.class,
                  () -> {
                    if (strided) {
                      CuVSMatrix.deviceBuilder(
                          failingResources, 1, columns, columns + 1, -1, CuVSMatrix.DataType.FLOAT);
                    } else {
                      CuVSMatrix.deviceBuilder(
                          failingResources, 1, columns, CuVSMatrix.DataType.FLOAT);
                    }
                  }));
        });
  }

  @Test
  public void testArrayFactoriesKeepExistingShortRowErrors() {
    // Cleanup must not change the error reported by the existing bulk-copy implementation.
    assertThrows(
        IndexOutOfBoundsException.class, () -> CuVSMatrix.ofArray(new float[][] {{1, 2}, {3}}));
    assertThrows(
        IndexOutOfBoundsException.class, () -> CuVSMatrix.ofArray(new int[][] {{1, 2}, {3}}));
    assertThrows(
        IndexOutOfBoundsException.class, () -> CuVSMatrix.ofArray(new byte[][] {{1, 2}, {3}}));
  }

  @Test
  public void testArrayFactoriesKeepTheFirstRowsWidth() {
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
  public void testHostBuilderKeepsStrideValidationErrors() {
    assertThrows(
        IllegalArgumentException.class,
        () -> CuVSMatrix.hostBuilder(2, 4, 3, -1, CuVSMatrix.DataType.FLOAT));
    assertThrows(
        UnsupportedOperationException.class,
        () -> CuVSMatrix.hostBuilder(2, 4, -1, 2, CuVSMatrix.DataType.FLOAT));
  }

  /** Real default conversion and destination allocation, with only the source copy made to fail. */
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
      long allocated = csvLong(header, lastSample, "workspace_total_alloc");
      assertTrue("the test did not allocate device memory", allocated > 0);
      assertEquals(
          "failed operation retained device memory",
          0,
          csvLong(header, lastSample, "workspace_current"));
      assertEquals(allocated, csvLong(header, lastSample, "workspace_total_freed"));
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

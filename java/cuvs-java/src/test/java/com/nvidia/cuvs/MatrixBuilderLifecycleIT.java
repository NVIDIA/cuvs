/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;

import java.lang.reflect.Proxy;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;

public class MatrixBuilderLifecycleIT {

  @Test
  public void testSuccessfulBuildTransfersOwnershipAfterBuilderClose() {
    AtomicInteger matrixCloses = new AtomicInteger();
    CuVSMatrix matrix = fakeMatrix(matrixCloses, null);
    FakeBuilder builder = new FakeBuilder(matrix, null, null);

    CuVSMatrix built = MatrixBuilderLifecycle.buildAndClose(builder);

    assertSame(matrix, built);
    assertEquals(1, builder.closeCalls.get());
    assertEquals(0, matrixCloses.get());
  }

  @Test
  public void testBuilderCloseFailureClosesTransferredMatrix() {
    IllegalStateException builderFailure = new IllegalStateException("builder close");
    AtomicInteger matrixCloses = new AtomicInteger();
    CuVSMatrix matrix = fakeMatrix(matrixCloses, null);
    FakeBuilder builder = new FakeBuilder(matrix, null, builderFailure);

    IllegalStateException thrown =
        assertThrows(
            IllegalStateException.class, () -> MatrixBuilderLifecycle.buildAndClose(builder));

    assertSame(builderFailure, thrown);
    assertEquals(1, builder.closeCalls.get());
    assertEquals(1, matrixCloses.get());
  }

  @Test
  public void testMatrixCloseFailureIsSuppressedOnBuilderCloseFailure() {
    IllegalStateException builderFailure = new IllegalStateException("builder close");
    IllegalArgumentException matrixFailure = new IllegalArgumentException("matrix close");
    AtomicInteger matrixCloses = new AtomicInteger();
    CuVSMatrix matrix = fakeMatrix(matrixCloses, matrixFailure);
    FakeBuilder builder = new FakeBuilder(matrix, null, builderFailure);

    IllegalStateException thrown =
        assertThrows(
            IllegalStateException.class, () -> MatrixBuilderLifecycle.buildAndClose(builder));

    assertSame(builderFailure, thrown);
    assertArrayEquals(new Throwable[] {matrixFailure}, thrown.getSuppressed());
    assertEquals(1, matrixCloses.get());
  }

  @Test
  public void testBuildFailureRemainsPrimaryWhenBuilderCloseAlsoFails() {
    IllegalArgumentException buildFailure = new IllegalArgumentException("build");
    IllegalStateException closeFailure = new IllegalStateException("builder close");
    FakeBuilder builder = new FakeBuilder(null, buildFailure, closeFailure);

    IllegalArgumentException thrown =
        assertThrows(
            IllegalArgumentException.class, () -> MatrixBuilderLifecycle.buildAndClose(builder));

    assertSame(buildFailure, thrown);
    assertArrayEquals(new Throwable[] {closeFailure}, thrown.getSuppressed());
    assertEquals(1, builder.closeCalls.get());
  }

  private static CuVSMatrix fakeMatrix(AtomicInteger closeCalls, RuntimeException closeFailure) {
    return (CuVSMatrix)
        Proxy.newProxyInstance(
            CuVSMatrix.class.getClassLoader(),
            new Class<?>[] {CuVSMatrix.class},
            (proxy, method, args) -> {
              if (method.getName().equals("close") && method.getParameterCount() == 0) {
                closeCalls.incrementAndGet();
                if (closeFailure != null) {
                  throw closeFailure;
                }
                return null;
              }
              throw new AssertionError("Unexpected matrix call: " + method);
            });
  }

  private static final class FakeBuilder implements CuVSMatrix.Builder<CuVSMatrix> {
    private final CuVSMatrix matrix;
    private final RuntimeException buildFailure;
    private final RuntimeException closeFailure;
    private final AtomicInteger closeCalls = new AtomicInteger();

    private FakeBuilder(
        CuVSMatrix matrix, RuntimeException buildFailure, RuntimeException closeFailure) {
      this.matrix = matrix;
      this.buildFailure = buildFailure;
      this.closeFailure = closeFailure;
    }

    @Override
    public void addVector(float[] vector) {}

    @Override
    public void addVector(byte[] vector) {}

    @Override
    public void addVector(int[] vector) {}

    @Override
    public void addVector(short[] vector) {}

    @Override
    public CuVSMatrix build() {
      if (buildFailure != null) {
        throw buildFailure;
      }
      return matrix;
    }

    @Override
    public void close() {
      closeCalls.incrementAndGet();
      if (closeFailure != null) {
        throw closeFailure;
      }
    }
  }
}

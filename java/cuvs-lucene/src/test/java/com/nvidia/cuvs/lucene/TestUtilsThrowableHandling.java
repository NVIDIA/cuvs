/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CuVSMatrix;
import java.io.IOException;
import java.lang.reflect.Proxy;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

public class TestUtilsThrowableHandling extends LuceneTestCase {

  @Test
  public void testHandleThrowableRethrowsIOExceptionUnchanged() {
    IOException exception = new IOException("I/O failure");

    IOException thrown = assertThrows(IOException.class, () -> Utils.handleThrowable(exception));

    assertSame(exception, thrown);
  }

  @Test
  public void testHandleThrowableRethrowsRuntimeExceptionUnchanged() {
    RuntimeException exception = new IllegalStateException("runtime failure");

    RuntimeException thrown =
        assertThrows(RuntimeException.class, () -> Utils.handleThrowable(exception));

    assertSame(exception, thrown);
  }

  @Test
  public void testHandleThrowableRethrowsErrorUnchanged() {
    Error error = new AssertionError("fatal failure");

    Error thrown = assertThrows(Error.class, () -> Utils.handleThrowable(error));

    assertSame(error, thrown);
  }

  @Test
  public void testHandleThrowableWrapsCheckedExceptionWithCause() {
    Exception exception = new Exception("checked failure");

    RuntimeException thrown =
        assertThrows(RuntimeException.class, () -> Utils.handleThrowable(exception));

    assertSame(exception, thrown.getCause());
  }

  @Test
  public void testMatrixBuilderCloseFailureClosesTransferredMatrix() {
    IllegalStateException builderFailure = new IllegalStateException("builder close");
    AtomicInteger matrixCloses = new AtomicInteger();
    CuVSMatrix matrix = fakeMatrix(matrixCloses, null);
    FakeMatrixBuilder builder = new FakeMatrixBuilder(matrix, builderFailure);

    IllegalStateException thrown =
        assertThrows(
            IllegalStateException.class,
            () -> MatrixBuilderLifecycle.build(builder, CuVSMatrix.Builder::build));

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
    FakeMatrixBuilder builder = new FakeMatrixBuilder(matrix, builderFailure);

    IllegalStateException thrown =
        assertThrows(
            IllegalStateException.class,
            () -> MatrixBuilderLifecycle.build(builder, CuVSMatrix.Builder::build));

    assertSame(builderFailure, thrown);
    assertArrayEquals(new Throwable[] {matrixFailure}, thrown.getSuppressed());
    assertEquals(1, matrixCloses.get());
  }

  @Test
  public void testMatrixOperationFailureRemainsPrimaryWhenBuilderCloseFails() {
    IOException operationFailure = new IOException("populate");
    IllegalStateException builderFailure = new IllegalStateException("builder close");
    FakeMatrixBuilder builder = new FakeMatrixBuilder(null, builderFailure);

    IOException thrown =
        assertThrows(
            IOException.class,
            () ->
                MatrixBuilderLifecycle.build(
                    builder,
                    ignored -> {
                      throw operationFailure;
                    }));

    assertSame(operationFailure, thrown);
    assertArrayEquals(new Throwable[] {builderFailure}, thrown.getSuppressed());
    assertEquals(1, builder.closeCalls.get());
  }

  @Test
  public void testOwnedIndexClosesUntransferredDatasetOnce() throws Exception {
    TrackingCloseable dataset = new TrackingCloseable(null);
    Utils.OwnedIndex<TrackingCloseable> owned = Utils.ownDataset(dataset);

    owned.close();
    owned.close();

    assertEquals(1, dataset.closeCount);
  }

  @Test
  public void testOwnedIndexDoesNotDirectlyCloseDatasetAfterSuccessfulIndexClose()
      throws Exception {
    TrackingCloseable dataset = new TrackingCloseable(null);
    TrackingCloseable index = new TrackingCloseable(null);
    Utils.OwnedIndex<TrackingCloseable> owned = Utils.ownDataset(dataset);
    owned.transferTo(index);

    owned.close();

    assertEquals(1, index.closeCount);
    assertEquals(0, dataset.closeCount);
  }

  @Test
  public void testOwnedIndexPreservesBodyAndNestedCleanupFailures() {
    IOException bodyFailure = new IOException("body");
    IOException indexCloseFailure = new IOException("index close");
    IOException datasetCloseFailure = new IOException("dataset close");
    TrackingCloseable dataset = new TrackingCloseable(datasetCloseFailure);
    TrackingCloseable index = new TrackingCloseable(indexCloseFailure);

    IOException thrown =
        assertThrows(
            IOException.class,
            () -> {
              try (Utils.OwnedIndex<TrackingCloseable> owned = Utils.ownDataset(dataset)) {
                owned.transferTo(index);
                throw bodyFailure;
              }
            });

    assertSame(bodyFailure, thrown);
    assertEquals(1, thrown.getSuppressed().length);
    assertSame(indexCloseFailure, thrown.getSuppressed()[0]);
    assertEquals(1, indexCloseFailure.getSuppressed().length);
    assertSame(datasetCloseFailure, indexCloseFailure.getSuppressed()[0]);
    assertEquals(1, index.closeCount);
    assertEquals(1, dataset.closeCount);
  }

  private static final class TrackingCloseable implements AutoCloseable {
    private final Exception failure;
    private int closeCount;

    private TrackingCloseable(Exception failure) {
      this.failure = failure;
    }

    @Override
    public void close() throws Exception {
      closeCount++;
      if (failure != null) {
        throw failure;
      }
    }
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

  private static final class FakeMatrixBuilder implements CuVSMatrix.Builder<CuVSMatrix> {
    private final CuVSMatrix matrix;
    private final RuntimeException closeFailure;
    private final AtomicInteger closeCalls = new AtomicInteger();

    private FakeMatrixBuilder(CuVSMatrix matrix, RuntimeException closeFailure) {
      this.matrix = matrix;
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

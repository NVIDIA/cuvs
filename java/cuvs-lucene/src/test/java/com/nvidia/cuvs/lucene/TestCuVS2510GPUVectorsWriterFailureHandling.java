/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CuVSDeviceMatrix;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.LibraryException;
import java.io.IOException;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

public class TestCuVS2510GPUVectorsWriterFailureHandling extends LuceneTestCase {

  @Test
  public void testDevicePreparationNativeFailureIsRecoverable() {
    LibraryException nativeFailure = new LibraryException("device upload");
    FakeDeviceBuilder builder = new FakeDeviceBuilder(null, null);

    Throwable thrown =
        assertThrows(
            Throwable.class,
            () ->
                CuVS2510GPUVectorsWriter.prepareCagraDataset(
                    () -> builder,
                    ignored -> {
                      throw nativeFailure;
                    }));

    assertTrue(thrown instanceof CuVS2510GPUVectorsWriter.RecoverableCagraConstructionException);
    assertSame(nativeFailure, thrown.getCause());
    assertEquals(0, nativeFailure.getSuppressed().length);
    assertEquals(1, builder.closeCalls.get());
  }

  @Test
  public void testDevicePreparationOperationAndCleanupFailureIsFatal() {
    LibraryException nativeFailure = new LibraryException("device upload");
    IllegalStateException cleanupFailure = new IllegalStateException("builder close");
    FakeDeviceBuilder builder = new FakeDeviceBuilder(null, cleanupFailure);

    Throwable thrown =
        assertThrows(
            Throwable.class,
            () ->
                CuVS2510GPUVectorsWriter.prepareCagraDataset(
                    () -> builder,
                    ignored -> {
                      throw nativeFailure;
                    }));

    assertSame(nativeFailure, thrown);
    assertArrayEquals(new Throwable[] {cleanupFailure}, thrown.getSuppressed());
    assertEquals(1, builder.closeCalls.get());
  }

  @Test
  public void testDevicePreparationNativeFactoryFailureIsFatal() {
    LibraryException factoryFailure = new LibraryException("builder factory");
    FakeDeviceBuilder builder = new FakeDeviceBuilder(null, null);

    Throwable thrown =
        assertThrows(
            Throwable.class,
            () ->
                CuVS2510GPUVectorsWriter.prepareCagraDataset(
                    () -> {
                      throw factoryFailure;
                    },
                    ignored -> fail("population must not run")));

    assertSame(factoryFailure, thrown);
    assertFalse(thrown instanceof CuVS2510GPUVectorsWriter.RecoverableCagraConstructionException);
    assertEquals(0, builder.closeCalls.get());
  }

  @Test
  public void testDevicePreparationTransfersOwnershipAfterBuilderCleanup() throws Throwable {
    AtomicInteger datasetCloses = new AtomicInteger();
    CuVSDeviceMatrix dataset = fakeDeviceMatrix(datasetCloses, null);
    FakeDeviceBuilder builder = new FakeDeviceBuilder(dataset, null);

    CuVSDeviceMatrix prepared =
        CuVS2510GPUVectorsWriter.prepareCagraDataset(() -> builder, ignored -> {});

    assertSame(dataset, prepared);
    assertEquals(1, builder.closeCalls.get());
    assertEquals(0, datasetCloses.get());
  }

  @Test
  public void testDevicePreparationBuilderCleanupFailureClosesTransferredDataset() {
    LibraryException cleanupFailure = new LibraryException("builder close");
    AtomicInteger datasetCloses = new AtomicInteger();
    CuVSDeviceMatrix dataset = fakeDeviceMatrix(datasetCloses, null);
    FakeDeviceBuilder builder = new FakeDeviceBuilder(dataset, cleanupFailure);

    Throwable thrown =
        assertThrows(
            Throwable.class,
            () -> CuVS2510GPUVectorsWriter.prepareCagraDataset(() -> builder, ignored -> {}));

    assertSame(cleanupFailure, thrown);
    assertEquals(1, builder.closeCalls.get());
    assertEquals(1, datasetCloses.get());
  }

  @Test
  public void testDevicePreparationDatasetCleanupFailureIsSuppressed() {
    LibraryException builderFailure = new LibraryException("builder close");
    IllegalStateException datasetFailure = new IllegalStateException("dataset close");
    AtomicInteger datasetCloses = new AtomicInteger();
    CuVSDeviceMatrix dataset = fakeDeviceMatrix(datasetCloses, datasetFailure);
    FakeDeviceBuilder builder = new FakeDeviceBuilder(dataset, builderFailure);

    Throwable thrown =
        assertThrows(
            Throwable.class,
            () -> CuVS2510GPUVectorsWriter.prepareCagraDataset(() -> builder, ignored -> {}));

    assertSame(builderFailure, thrown);
    assertArrayEquals(new Throwable[] {datasetFailure}, thrown.getSuppressed());
    assertEquals(1, builder.closeCalls.get());
    assertEquals(1, datasetCloses.get());
  }

  @Test
  public void testDevicePreparationPreservesNonNativeFailures() {
    IOException checkedFailure = new IOException("checked operation");
    AssertionError error = new AssertionError("operation error");

    for (Throwable expected : List.of(checkedFailure, error)) {
      FakeDeviceBuilder builder = new FakeDeviceBuilder(null, null);
      Throwable thrown =
          assertThrows(
              Throwable.class,
              () ->
                  CuVS2510GPUVectorsWriter.prepareCagraDataset(
                      () -> builder,
                      ignored -> {
                        throw expected;
                      }));

      assertSame(expected, thrown);
      assertEquals(1, builder.closeCalls.get());
    }
  }

  @Test
  public void testNativeConstructionFailureIsRecoverableBeforePersistence() {
    LibraryException nativeFailure = new LibraryException("native construction");

    Throwable classified =
        CuVS2510GPUVectorsWriter.classifyCagraWriteFailure(nativeFailure, false, false);

    assertTrue(
        classified instanceof CuVS2510GPUVectorsWriter.RecoverableCagraConstructionException);
    assertSame(nativeFailure, classified.getCause());
  }

  @Test
  public void testArbitraryConstructionFailureIsNotRecoverable() {
    RuntimeException programmingFailure = new IllegalStateException("programming failure");

    Throwable classified =
        CuVS2510GPUVectorsWriter.classifyCagraWriteFailure(programmingFailure, false, false);

    assertSame(programmingFailure, classified);
  }

  @Test
  public void testNativeFailureIsNotRecoverableAfterPersistenceStarts() {
    LibraryException nativeFailure = new LibraryException("serialization");

    Throwable classified =
        CuVS2510GPUVectorsWriter.classifyCagraWriteFailure(nativeFailure, true, false);

    assertSame(nativeFailure, classified);
  }

  @Test
  public void testCleanupFailureDisablesFallback() {
    LibraryException nativeFailure = new LibraryException("native construction");
    nativeFailure.addSuppressed(new IOException("cleanup"));

    Throwable classified =
        CuVS2510GPUVectorsWriter.classifyCagraWriteFailure(nativeFailure, false, true);

    assertSame(nativeFailure, classified);
  }

  @Test
  public void testSuppressedFailureDisablesFallback() {
    LibraryException nativeFailure = new LibraryException("native construction");
    nativeFailure.addSuppressed(new IOException("cleanup"));

    Throwable classified =
        CuVS2510GPUVectorsWriter.classifyCagraWriteFailure(nativeFailure, false, false);

    assertSame(nativeFailure, classified);
  }

  @Test
  public void testPersistenceBoundaryIsMonotonic() {
    CuVS2510GPUVectorsWriter.CagraWriteContext context =
        new CuVS2510GPUVectorsWriter.CagraWriteContext();

    assertFalse(context.persistenceStarted());
    context.beginPersistence();
    assertTrue(context.persistenceStarted());
  }

  @Test
  public void testChangedOutputPositionRejectsFallback() {
    Throwable failure = new LibraryException("native construction");

    IOException thrown =
        assertThrows(
            IOException.class,
            () -> CuVS2510GPUVectorsWriter.ensureFallbackOutputUnchanged(10L, 11L, failure));

    assertSame(failure, thrown.getCause());
    assertTrue(thrown.getMessage().contains("changed from 10 to 11"));
  }

  @Test
  public void testUnchangedOutputPositionAllowsFallback() throws IOException {
    CuVS2510GPUVectorsWriter.ensureFallbackOutputUnchanged(
        10L, 10L, new LibraryException("native construction"));
  }

  @Test
  public void testCagraResourcesCloseInDependencyOrder() {
    List<String> closeOrder = new ArrayList<>();
    AtomicInteger directDatasetCloses = new AtomicInteger();

    Throwable failure =
        CuVS2510GPUVectorsWriter.closeCagraResources(
            () -> closeOrder.add("index"),
            directDatasetCloses::incrementAndGet,
            () -> closeOrder.add("padded-dataset"));

    assertNull(failure);
    assertEquals(List.of("index", "padded-dataset"), closeOrder);
    assertEquals(0, directDatasetCloses.get());
  }

  @Test
  public void testBodyAndCleanupFailuresPreserveOrderAndSuppression() {
    List<String> closeOrder = new ArrayList<>();
    IOException bodyFailure = new IOException("serialize");
    RuntimeException indexFailure = new RuntimeException("index close");
    RuntimeException datasetFailure = new RuntimeException("dataset close");
    RuntimeException paddedFailure = new RuntimeException("padded close");

    Throwable cleanupFailure =
        CuVS2510GPUVectorsWriter.closeCagraResources(
            failingCloseable("index", closeOrder, indexFailure),
            failingCloseable("dataset", closeOrder, datasetFailure),
            failingCloseable("padded-dataset", closeOrder, paddedFailure));
    Throwable combined =
        CuVS2510GPUVectorsWriter.combineOperationAndCleanupFailures(bodyFailure, cleanupFailure);

    assertSame(bodyFailure, combined);
    assertEquals(List.of("index", "dataset", "padded-dataset"), closeOrder);
    assertArrayEquals(new Throwable[] {indexFailure}, bodyFailure.getSuppressed());
    assertArrayEquals(
        new Throwable[] {datasetFailure, paddedFailure}, indexFailure.getSuppressed());
  }

  private static AutoCloseable failingCloseable(
      String name, List<String> closeOrder, RuntimeException failure) {
    return () -> {
      closeOrder.add(name);
      throw failure;
    };
  }

  private static CuVSDeviceMatrix fakeDeviceMatrix(
      AtomicInteger closeCalls, RuntimeException closeFailure) {
    return (CuVSDeviceMatrix)
        Proxy.newProxyInstance(
            CuVSDeviceMatrix.class.getClassLoader(),
            new Class<?>[] {CuVSDeviceMatrix.class},
            (proxy, method, args) -> {
              if (method.getName().equals("close") && method.getParameterCount() == 0) {
                closeCalls.incrementAndGet();
                if (closeFailure != null) {
                  throw closeFailure;
                }
                return null;
              }
              throw new AssertionError("Unexpected device-matrix call: " + method);
            });
  }

  private static final class FakeDeviceBuilder implements CuVSMatrix.Builder<CuVSDeviceMatrix> {
    private final CuVSDeviceMatrix dataset;
    private final RuntimeException closeFailure;
    private final AtomicInteger closeCalls = new AtomicInteger();

    private FakeDeviceBuilder(CuVSDeviceMatrix dataset, RuntimeException closeFailure) {
      this.dataset = dataset;
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
    public CuVSDeviceMatrix build() {
      return dataset;
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

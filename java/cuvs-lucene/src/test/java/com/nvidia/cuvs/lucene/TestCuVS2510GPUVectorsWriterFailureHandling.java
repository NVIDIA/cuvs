/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.LibraryException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

public class TestCuVS2510GPUVectorsWriterFailureHandling extends LuceneTestCase {

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
            () -> closeOrder.add("padded-dataset"),
            () -> closeOrder.add("device-vectors"));

    assertNull(failure);
    assertEquals(List.of("index", "padded-dataset", "device-vectors"), closeOrder);
    assertEquals(0, directDatasetCloses.get());
  }

  @Test
  public void testBodyAndCleanupFailuresPreserveOrderAndSuppression() {
    List<String> closeOrder = new ArrayList<>();
    IOException bodyFailure = new IOException("serialize");
    RuntimeException indexFailure = new RuntimeException("index close");
    RuntimeException datasetFailure = new RuntimeException("dataset close");
    RuntimeException paddedFailure = new RuntimeException("padded close");
    RuntimeException deviceFailure = new RuntimeException("device close");

    Throwable cleanupFailure =
        CuVS2510GPUVectorsWriter.closeCagraResources(
            failingCloseable("index", closeOrder, indexFailure),
            failingCloseable("dataset", closeOrder, datasetFailure),
            failingCloseable("padded-dataset", closeOrder, paddedFailure),
            failingCloseable("device-vectors", closeOrder, deviceFailure));
    Throwable combined =
        CuVS2510GPUVectorsWriter.combineOperationAndCleanupFailures(bodyFailure, cleanupFailure);

    assertSame(bodyFailure, combined);
    assertEquals(List.of("index", "dataset", "padded-dataset", "device-vectors"), closeOrder);
    assertArrayEquals(new Throwable[] {indexFailure}, bodyFailure.getSuppressed());
    assertArrayEquals(
        new Throwable[] {datasetFailure, paddedFailure, deviceFailure},
        indexFailure.getSuppressed());
  }

  private static AutoCloseable failingCloseable(
      String name, List<String> closeOrder, RuntimeException failure) {
    return () -> {
      closeOrder.add(name);
      throw failure;
    };
  }
}

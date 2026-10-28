/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CuVSHostMatrix;
import com.nvidia.cuvs.CuVSMatrix;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.InfoStream;

public class TestHostInputMemory extends LuceneTestCase {
  public void testCompactFloatAndQuantizedPayloadSizes() {
    assertEquals(5120, HostInputMemory.payloadBytes(10, 128, CuVSMatrix.DataType.FLOAT));
    assertEquals(1280, HostInputMemory.payloadBytes(10, 128, CuVSMatrix.DataType.BYTE));
    // A packed binary vector with 129 dimensions needs 17 bytes, not 16.
    assertEquals(170, HostInputMemory.payloadBytes(10, 17, CuVSMatrix.DataType.BYTE));
    assertEquals(0, HostInputMemory.payloadBytes(0, 128, CuVSMatrix.DataType.FLOAT));
  }

  public void testPayloadArithmeticRejectsOverflowAndNegativeDimensions() {
    expectThrows(
        ArithmeticException.class,
        () -> HostInputMemory.payloadBytes(Long.MAX_VALUE, 2, CuVSMatrix.DataType.BYTE));
    expectThrows(
        ArithmeticException.class,
        () -> HostInputMemory.payloadBytes(Long.MAX_VALUE / 2, 1, CuVSMatrix.DataType.FLOAT));
    expectThrows(
        IllegalArgumentException.class,
        () -> HostInputMemory.payloadBytes(-1, 128, CuVSMatrix.DataType.FLOAT));
    expectThrows(
        IllegalArgumentException.class,
        () -> HostInputMemory.payloadBytes(1, -1, CuVSMatrix.DataType.FLOAT));
  }

  public void testPayloadAboveTwoGiBIsCountedUntilCleanupWithoutAllocatingIt() throws Exception {
    HostInputMemory memory = newMemory();
    long baseline = memory.ramBytesUsed();
    long threeGiB = HostInputMemory.payloadBytes(3L * 1024 * 1024, 256, CuVSMatrix.DataType.FLOAT);
    assertEquals(3L * 1024 * 1024 * 1024, threeGiB);

    // Repeat with a different field size to catch stale or cumulative accounting.
    for (long bytes : new long[] {threeGiB, 512}) {
      TrackingBuilder builder =
          new TrackingBuilder(() -> assertEquals(baseline + bytes, memory.ramBytesUsed()));
      memory.withAllocation(
          "embedding",
          bytes,
          () -> {
            assertEquals(baseline, memory.ramBytesUsed());
            return builder;
          },
          allocated -> {
            assertSame(builder, allocated);
            assertEquals(baseline + bytes, memory.ramBytesUsed());
            allocated.addVector(new float[] {1f});
          });
      assertEquals(1, builder.addCalls);
      assertEquals(1, builder.closeCalls);
      assertEquals(baseline, memory.ramBytesUsed());
    }
  }

  public void testFailedAllocationNeverPublishesPayload() {
    HostInputMemory memory = newMemory();
    long baseline = memory.ramBytesUsed();
    OutOfMemoryError allocationFailure =
        new OutOfMemoryError("simulated native allocation failure");

    OutOfMemoryError thrown =
        expectThrows(
            OutOfMemoryError.class,
            () ->
                memory.withAllocation(
                    "embedding",
                    512,
                    () -> {
                      throw allocationFailure;
                    },
                    builder -> fail("No builder exists after failed allocation")));

    assertSame(allocationFailure, thrown);
    assertEquals(baseline, memory.ramBytesUsed());
  }

  public void testBuildFailureRemainsPrimaryWhenBuilderCleanupFails() {
    HostInputMemory memory = newMemory();
    long baseline = memory.ramBytesUsed();
    IllegalStateException closeFailure = new IllegalStateException("builder cleanup failed");
    TrackingBuilder builder =
        new TrackingBuilder(
            () -> {
              throw closeFailure;
            });

    IllegalStateException thrown =
        expectThrows(
            IllegalStateException.class,
            () ->
                memory.withAllocation(
                    "embedding", 512, () -> builder, allocated -> allocated.build()));

    assertSame(builder.buildFailure, thrown);
    assertArrayEquals(new Throwable[] {closeFailure}, thrown.getSuppressed());
    assertEquals(1, builder.closeCalls);
    assertEquals(baseline, memory.ramBytesUsed());
  }

  public void testDiagnosticFailureStillClosesAllocatedBuilder() {
    IllegalStateException diagnosticFailure = new IllegalStateException("diagnostic sink failed");
    InfoStream failingLog =
        new InfoStream() {
          @Override
          public void message(String component, String message) {
            throw diagnosticFailure;
          }

          @Override
          public boolean isEnabled(String component) {
            return true;
          }

          @Override
          public void close() {}
        };
    HostInputMemory memory = new HostInputMemory(failingLog, "test", "_0");
    long baseline = memory.ramBytesUsed();
    TrackingBuilder builder = new TrackingBuilder(() -> {});

    IllegalStateException thrown =
        expectThrows(
            IllegalStateException.class,
            () ->
                memory.withAllocation(
                    "embedding",
                    512,
                    () -> builder,
                    allocated -> fail("Diagnostic failure prevents population")));

    assertSame(diagnosticFailure, thrown);
    assertEquals(1, builder.closeCalls);
    assertEquals(baseline, memory.ramBytesUsed());
  }

  private static HostInputMemory newMemory() {
    return new HostInputMemory(InfoStream.NO_OUTPUT, "test", "_0");
  }

  /** Only supplies the builder lifecycle; no native allocation or successful matrix build occurs. */
  private static final class TrackingBuilder implements CuVSMatrix.Builder<CuVSHostMatrix> {
    private final Runnable onClose;
    private final IllegalStateException buildFailure =
        new IllegalStateException("matrix build failed");
    private int addCalls;
    private int closeCalls;

    TrackingBuilder(Runnable onClose) {
      this.onClose = onClose;
    }

    @Override
    public void addVector(float[] vector) {
      addCalls++;
    }

    @Override
    public void addVector(byte[] vector) {
      throw new AssertionError("unexpected byte vector");
    }

    @Override
    public void addVector(int[] vector) {
      throw new AssertionError("unexpected int vector");
    }

    @Override
    public void addVector(short[] vector) {
      throw new AssertionError("unexpected short vector");
    }

    @Override
    public CuVSHostMatrix build() {
      throw buildFailure;
    }

    @Override
    public void close() {
      closeCalls++;
      onClose.run();
    }
  }
}

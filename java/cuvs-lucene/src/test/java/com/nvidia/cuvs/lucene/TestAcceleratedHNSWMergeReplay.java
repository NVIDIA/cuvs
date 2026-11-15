/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CuVSHostMatrix;
import com.nvidia.cuvs.CuVSMatrix;
import java.io.IOException;
import java.lang.reflect.Proxy;
import java.util.List;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.InfoStream;

public class TestAcceleratedHNSWMergeReplay extends LuceneTestCase {

  public void testUnderflowClosesBuilderBeforeBuild() {
    TrackingBuilder builder = new TrackingBuilder();
    FloatVectorValues values = FloatVectorValues.fromFloats(List.of(new float[] {1f}), 1);

    IOException failure =
        expectThrows(IOException.class, () -> replayWithinAccountingScope(values, 2, builder));

    assertEquals(
        "Merged vector count changed between passes: expected 2, observed 1", failure.getMessage());
    assertEquals(1, builder.addCalls);
    assertEquals(0, builder.buildCalls);
    assertEquals(1, builder.closeCalls);
  }

  public void testOverflowClosesBuilderBeforeBuild() {
    TrackingBuilder builder = new TrackingBuilder();
    FloatVectorValues values =
        FloatVectorValues.fromFloats(List.of(new float[] {1f}, new float[] {2f}), 1);

    IOException failure =
        expectThrows(IOException.class, () -> replayWithinAccountingScope(values, 1, builder));

    assertEquals(
        "Merged vector count changed between passes: expected 1, observed at least 2",
        failure.getMessage());
    assertEquals(1, builder.addCalls);
    assertEquals(0, builder.buildCalls);
    assertEquals(1, builder.closeCalls);
  }

  public void testReplayFailureRemainsPrimaryWhenBuilderCloseFails() {
    IOException replayFailure = new IOException("vector read failed");
    IllegalStateException closeFailure = new IllegalStateException("builder close failed");
    TrackingBuilder builder = new TrackingBuilder(closeFailure);
    FloatVectorValues values = failingValues(replayFailure);

    IOException thrown =
        expectThrows(IOException.class, () -> replayWithinAccountingScope(values, 1, builder));

    assertSame(replayFailure, thrown);
    assertArrayEquals(new Throwable[] {closeFailure}, thrown.getSuppressed());
    assertEquals(0, builder.addCalls);
    assertEquals(0, builder.buildCalls);
    assertEquals(1, builder.closeCalls);
  }

  public void testSuccessfulReplayClosesBuilderExactlyOnce() throws IOException {
    TrackingBuilder builder = new TrackingBuilder();
    int[] datasetCloses = {0};
    // Replay only passes this returned handle to its owner. No native allocation is needed.
    builder.builtMatrix =
        (CuVSHostMatrix)
            Proxy.newProxyInstance(
                CuVSHostMatrix.class.getClassLoader(),
                new Class<?>[] {CuVSHostMatrix.class},
                (proxy, method, arguments) -> {
                  if (method.getName().equals("close")) {
                    datasetCloses[0]++;
                    return null;
                  }
                  throw new AssertionError(
                      "Replay must not inspect the returned matrix: " + method.getName());
                });
    FloatVectorValues values = FloatVectorValues.fromFloats(List.of(new float[] {1f}), 1);

    replayWithinAccountingScope(values, 1, builder);

    assertEquals(1, builder.addCalls);
    assertEquals(1, builder.buildCalls);
    assertEquals(1, builder.closeCalls);
    assertEquals(1, datasetCloses[0]);
  }

  private static void replayWithinAccountingScope(
      FloatVectorValues values, int expectedSize, TrackingBuilder builder) throws IOException {
    HostInputMemory memory = new HostInputMemory(InfoStream.NO_OUTPUT, "test", "_0");
    long baseline = memory.ramBytesUsed();
    long payloadBytes = (long) expectedSize * values.dimension() * Float.BYTES;
    try {
      memory.withAllocation(
          "embedding",
          payloadBytes,
          () -> builder,
          allocated -> {
            try (CuVSHostMatrix dataset =
                Lucene99AcceleratedHNSWVectorsWriter.buildMergedDataset(
                    values, expectedSize, allocated)) {
              assertSame(builder.builtMatrix, dataset);
              assertEquals(baseline + payloadBytes, memory.ramBytesUsed());
            } catch (Exception e) {
              throw Utils.handleThrowable(e);
            }
          });
    } finally {
      assertEquals(baseline, memory.ramBytesUsed());
    }
  }

  private static FloatVectorValues failingValues(IOException failure) {
    FloatVectorValues delegate = FloatVectorValues.fromFloats(List.of(new float[] {1f}), 1);
    return new FloatVectorValues() {
      @Override
      public int dimension() {
        return 1;
      }

      @Override
      public int size() {
        return 1;
      }

      @Override
      public float[] vectorValue(int ord) throws IOException {
        throw failure;
      }

      @Override
      public FloatVectorValues copy() {
        return this;
      }

      @Override
      public KnnVectorValues.DocIndexIterator iterator() {
        return delegate.iterator();
      }
    };
  }

  private static final class TrackingBuilder implements CuVSMatrix.Builder<CuVSHostMatrix> {
    private final RuntimeException closeFailure;
    private int addCalls;
    private int buildCalls;
    private int closeCalls;
    private CuVSHostMatrix builtMatrix;

    private TrackingBuilder() {
      this(null);
    }

    private TrackingBuilder(RuntimeException closeFailure) {
      this.closeFailure = closeFailure;
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
      buildCalls++;
      if (builtMatrix == null) {
        throw new AssertionError("build must not be called");
      }
      return builtMatrix;
    }

    @Override
    public void close() {
      closeCalls++;
      if (closeCalls != 1) {
        throw new AssertionError("Builder must have exactly one cleanup owner");
      }
      if (closeFailure != null) {
        throw closeFailure;
      }
    }
  }
}

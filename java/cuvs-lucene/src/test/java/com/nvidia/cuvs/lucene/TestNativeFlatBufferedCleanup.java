/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSResources;
import java.io.Closeable;
import java.io.IOException;
import java.lang.reflect.Proxy;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.tests.store.MockDirectoryWrapper;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.junit.Test;

public class TestNativeFlatBufferedCleanup extends LuceneTestCase {

  @Test
  public void testThreadResourcesCloseWhenOutputCloseFails() {
    AtomicBoolean resourcesClosed = new AtomicBoolean();
    CuVSResources resources = resourcesThatRecordClose(resourcesClosed);
    ThreadLocalCuVSResourcesProvider.setCuVSResourcesInstance(resources);
    Closeable failingOutput =
        () -> {
          throw new IOException("output close failed");
        };

    IOException failure =
        expectThrows(
            IOException.class,
            () -> NativeFlatBufferedHNSWVectorsWriter.closeOutputsAndResources(failingOutput));

    assertEquals("output close failed", failure.getMessage());
    assertTrue("cuVS resources were not closed", resourcesClosed.get());
  }

  @Test
  public void testBorrowedWriterConstructorFailureClosesThreadResource() throws Exception {
    AtomicInteger closeCount = new AtomicInteger();
    AtomicReference<Throwable> constructorFailure = new AtomicReference<>();
    try (MockDirectoryWrapper directory = newMockDirectory()) {
      SegmentInfo segmentInfo =
          new SegmentInfo(
              directory,
              Version.LATEST,
              Version.LATEST,
              "_0",
              1,
              false,
              false,
              Codec.getDefault(),
              Map.of(),
              StringHelper.randomId(),
              Map.of(),
              null);
      SegmentWriteState state =
          new SegmentWriteState(
              InfoStream.NO_OUTPUT,
              directory,
              segmentInfo,
              new FieldInfos(new FieldInfo[0]),
              null,
              IOContext.DEFAULT);

      Thread constructorThread =
          new Thread(
              () -> {
                ThreadLocalCuVSResourcesProvider.setCuVSResourcesInstance(
                    resourcesThatCountsClose(closeCount));
                try {
                  new BorrowedDatasetHnswVectorsWriter(
                      state,
                      new AcceleratedHNSWParams.Builder().build(),
                      BulkIndexingContext.nativeBuffered(1, new CagraHnswBuildMetrics()));
                  constructorFailure.set(new AssertionError("constructor unexpectedly succeeded"));
                } catch (Throwable failure) {
                  constructorFailure.set(failure);
                }
              },
              "borrowed-writer-constructor-failure");
      constructorThread.start();
      constructorThread.join();

      assertNotNull(constructorFailure.get());
      assertTrue(constructorFailure.get() instanceof IllegalArgumentException);
      assertTrue(
          constructorFailure.get().getMessage(),
          constructorFailure.get().getMessage().contains("requires a mapped dataset"));
      assertEquals(1, closeCount.get());
      assertEquals(0, directory.getFileHandleCount());
    }
  }

  @Test
  public void testOwnedDatasetCloseFailureIsSuppressedOntoWriteFailure() {
    CuVSMatrix dataset =
        (CuVSMatrix)
            Proxy.newProxyInstance(
                CuVSMatrix.class.getClassLoader(),
                new Class<?>[] {CuVSMatrix.class},
                (proxy, method, arguments) -> {
                  if (method.getName().equals("close")) {
                    throw new IllegalStateException("dataset close failed");
                  }
                  throw new UnsupportedOperationException(method.getName());
                });

    IOException failure =
        expectThrows(
            IOException.class,
            () ->
                AcceleratedHnswGraphOutput.withOwnedDataset(
                    dataset,
                    () -> {
                      throw new IOException("graph write failed");
                    }));

    assertEquals("graph write failed", failure.getMessage());
    assertEquals(1, failure.getSuppressed().length);
    assertEquals("dataset close failed", failure.getSuppressed()[0].getMessage());
  }

  private static CuVSResources resourcesThatRecordClose(AtomicBoolean closed) {
    return (CuVSResources)
        Proxy.newProxyInstance(
            CuVSResources.class.getClassLoader(),
            new Class<?>[] {CuVSResources.class},
            (proxy, method, arguments) -> {
              if (method.getName().equals("close")) {
                closed.set(true);
                return null;
              }
              throw new UnsupportedOperationException(method.getName());
            });
  }

  private static CuVSResources resourcesThatCountsClose(AtomicInteger closeCount) {
    return (CuVSResources)
        Proxy.newProxyInstance(
            CuVSResources.class.getClassLoader(),
            new Class<?>[] {CuVSResources.class},
            (proxy, method, arguments) -> {
              if (method.getName().equals("close")) {
                closeCount.incrementAndGet();
                return null;
              }
              throw new UnsupportedOperationException(method.getName());
            });
  }
}

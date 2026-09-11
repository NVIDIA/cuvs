/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CuVSResources;
import java.io.Closeable;
import java.io.IOException;
import java.lang.reflect.Proxy;
import java.util.concurrent.atomic.AtomicBoolean;
import org.apache.lucene.tests.util.LuceneTestCase;
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
}

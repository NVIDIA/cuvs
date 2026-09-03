/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import org.junit.Test;

public class TestLucene102FormatConstruction {

  private static final String FORMAT_NAME = "Lucene102BinaryQuantizedVectorsFormat";

  @Test
  public void testMissingProviderCapabilityIsUnsupported() {
    ClassNotFoundException missing = new ClassNotFoundException("missing Lucene102 provider");

    UnsupportedOperationException thrown =
        assertThrows(
            UnsupportedOperationException.class,
            () ->
                LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.constructLucene102Format(
                    FORMAT_NAME,
                    () -> {
                      throw missing;
                    }));

    assertSame(missing, thrown.getCause());
    assertTrue(thrown.getMessage().contains(FORMAT_NAME));
  }

  @Test
  public void testConstructorTargetIOExceptionIsRethrown() {
    IOException failure = new IOException("constructor I/O failure");

    IOException thrown =
        assertThrows(
            IOException.class,
            () ->
                LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.constructLucene102Format(
                    FORMAT_NAME, () -> invokeFailingConstructor(failure)));

    assertSame(failure, thrown);
  }

  @Test
  public void testConstructorTargetRuntimeExceptionIsRethrown() {
    IllegalArgumentException failure = new IllegalArgumentException("invalid arguments");

    IllegalArgumentException thrown =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.constructLucene102Format(
                    FORMAT_NAME, () -> invokeFailingConstructor(failure)));

    assertSame(failure, thrown);
  }

  @Test
  public void testConstructorTargetErrorIsRethrown() {
    AssertionError failure = new AssertionError("constructor error");

    AssertionError thrown =
        assertThrows(
            AssertionError.class,
            () ->
                LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.constructLucene102Format(
                    FORMAT_NAME, () -> invokeFailingConstructor(failure)));

    assertSame(failure, thrown);
  }

  @Test
  public void testConstructorTargetClassNotFoundIsConstructionFailure() {
    ClassNotFoundException failure = new ClassNotFoundException("failure inside constructor");

    IllegalStateException thrown =
        assertThrows(
            IllegalStateException.class,
            () ->
                LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.constructLucene102Format(
                    FORMAT_NAME, () -> invokeFailingConstructor(failure)));

    assertSame(failure, thrown.getCause());
    assertTrue(thrown.getMessage().contains("Unable to construct " + FORMAT_NAME));
  }

  @Test
  public void testCheckedReflectionFailureHasConstructionContext() {
    NoSuchMethodException failure = new NoSuchMethodException("missing constructor");

    IllegalStateException thrown =
        assertThrows(
            IllegalStateException.class,
            () ->
                LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.constructLucene102Format(
                    FORMAT_NAME,
                    () -> {
                      throw failure;
                    }));

    assertSame(failure, thrown.getCause());
    assertTrue(thrown.getMessage().contains("Unable to construct " + FORMAT_NAME));
  }

  private static Object invokeFailingConstructor(Throwable failure) throws Exception {
    return FailingConstructor.class.getDeclaredConstructor(Throwable.class).newInstance(failure);
  }

  private static final class FailingConstructor {
    private FailingConstructor(Throwable failure) throws Exception {
      if (failure instanceof Exception exception) {
        throw exception;
      }
      if (failure instanceof Error error) {
        throw error;
      }
      throw new AssertionError("Unexpected throwable", failure);
    }
  }
}

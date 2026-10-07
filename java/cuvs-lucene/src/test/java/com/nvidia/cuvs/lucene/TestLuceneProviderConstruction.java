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

/** Verifies how {@link LuceneProvider} classifies reflective construction failures. */
public class TestLuceneProviderConstruction {

  private static final String COMPONENT_NAME = "ThrowingConstructor";

  @Test
  public void testMissingProviderClassNamesBothAttemptedPackages() {
    ClassNotFoundException thrown =
        assertThrows(ClassNotFoundException.class, () -> LuceneProvider.getInstance("0"));

    assertTrue(thrown.getMessage().contains("org.apache.lucene.codecs.lucene0"));
    assertTrue(thrown.getMessage().contains("org.apache.lucene.backward_codecs.lucene0"));
    assertTrue(thrown.getSuppressed().length >= 2);
  }

  @Test
  public void testIOExceptionThrownByConstructorIsRethrownUnchanged() {
    IOException failure = new IOException("constructor I/O failure");

    IOException thrown =
        assertThrows(IOException.class, () -> invokeConstructorThatThrows(failure));

    assertSame(failure, thrown);
  }

  @Test
  public void testRuntimeExceptionThrownByConstructorIsRethrownUnchanged() {
    IllegalArgumentException failure = new IllegalArgumentException("invalid arguments");

    IllegalArgumentException thrown =
        assertThrows(IllegalArgumentException.class, () -> invokeConstructorThatThrows(failure));

    assertSame(failure, thrown);
  }

  @Test
  public void testErrorThrownByConstructorIsRethrownUnchanged() {
    AssertionError failure = new AssertionError("constructor error");

    AssertionError thrown =
        assertThrows(AssertionError.class, () -> invokeConstructorThatThrows(failure));

    assertSame(failure, thrown);
  }

  @Test
  public void testOtherCheckedExceptionBecomesConstructionFailureWithOriginalCause() {
    ClassNotFoundException failure = new ClassNotFoundException("failure inside constructor");

    IllegalStateException thrown =
        assertThrows(IllegalStateException.class, () -> invokeConstructorThatThrows(failure));

    assertSame(failure, thrown.getCause());
    assertTrue(thrown.getMessage().contains("Unable to initialize " + COMPONENT_NAME));
  }

  private static Object invokeConstructorThatThrows(Throwable failure) throws Exception {
    return LuceneProvider.invokeConstructor(
        COMPONENT_NAME, ThrowingConstructor.class.getConstructor(Throwable.class), failure);
  }

  public static final class ThrowingConstructor {
    public ThrowingConstructor(Throwable failure) throws Exception {
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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.spi;

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;

import com.nvidia.cuvs.CuVSTestCase;
import com.nvidia.cuvs.LibraryException;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;
import java.util.function.Supplier;
import org.junit.Before;
import org.junit.Test;

public class CuVSProviderIT extends CuVSTestCase {

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxSupportedArch());
    // Clear sysprop from previous runs/command line
    System.clearProperty("cuvs.max_version");
  }

  @Test
  public void testSameVersionCheck() {
    try {
      checkCuVSVersionMatching("25.12.0", 25, 12, 0);
    } catch (ProviderInitializationException e) {
      throw new AssertionError(e);
    }
  }

  @Test
  public void testPatchLevelNotConsidered() {
    try {
      checkCuVSVersionMatching("25.12.0", 25, 12, 1);
      checkCuVSVersionMatching("25.12.1", 25, 12, 0);
      checkCuVSVersionMatching("25.12", 25, 12, 0);
    } catch (ProviderInitializationException e) {
      throw new AssertionError(e);
    }
  }

  @Test
  public void testInvalidVersionsNotConsidered() {
    try {
      checkCuVSVersionMatching("abc", 25, 12, 0);
      checkCuVSVersionMatching("0.0.0", 25, 12, 0);
    } catch (ProviderInitializationException e) {
      throw new AssertionError(e);
    }
  }

  @Test
  public void testPastVersionCheck() {
    var ex =
        assertThrows(
            ProviderInitializationException.class,
            () -> checkCuVSVersionMatching("25.12.0", 25, 10, 0));

    assertEquals(
        """
        Version mismatch: outdated libcuvs_c (libcuvs_c [25.10.0], cuvs-java version [25.12.0]). \
        Please upgrade your libcuvs_c installation to match at lease the cuvs-java version.\
        """,
        ex.getMessage());
  }

  @Test
  public void testSupportedFutureVersionCheck() {
    try {
      checkCuVSVersionMatching("25.12.0", 26, 2, 0);
      checkCuVSVersionMatching("26.02.0", 26, 8, 0);
      checkCuVSVersionMatching("26.02.0", 26, 8, 1);
    } catch (ProviderInitializationException e) {
      throw new AssertionError(e);
    }
  }

  @Test
  public void testUnsupportedFutureVersionCheck() {
    var ex =
        assertThrows(
            ProviderInitializationException.class,
            () -> checkCuVSVersionMatching("25.12.0", 26, 10, 0));

    assertEquals(
        """
        Version mismatch: unsupported libcuvs_c (libcuvs_c [26.10.0], cuvs-java version [25.12.0]). \
        Please upgrade your software, or install a previous version of libcuvs_c.\
        """,
        ex.getMessage());
  }

  @Test
  public void testMaxVersionOverride() {
    try {
      checkCuVSVersionMatching("25.12.0", 26, 4, 0);
      System.setProperty("cuvs.max_version", "26.02.0");

      var ex =
          assertThrows(
              ProviderInitializationException.class,
              () -> checkCuVSVersionMatching("25.12.0", 26, 4, 0));

      assertEquals(
          """
          Version mismatch: unsupported libcuvs_c (libcuvs_c [26.04.0], cuvs-java version [25.12.0]). \
          Please upgrade your software, or install a previous version of libcuvs_c.\
          """,
          ex.getMessage());
      System.setProperty("cuvs.max_version", "26.12.0");
      checkCuVSVersionMatching("25.12.0", 26, 12, 0);
    } catch (ProviderInitializationException e) {
      throw new AssertionError(e);
    }
  }

  @Test
  public void testDeviceMatrixBuilderDoesNotAllocateWhenStreamAcquisitionFails() {
    LibraryException streamFailure = new LibraryException("stream acquisition");
    AtomicInteger streamCalls = new AtomicInteger();
    AtomicInteger builderCalls = new AtomicInteger();

    LibraryException thrown =
        assertThrows(
            LibraryException.class,
            () ->
                createDeviceMatrixBuilder(
                    () -> {
                      streamCalls.incrementAndGet();
                      throw streamFailure;
                    },
                    ignored -> {
                      builderCalls.incrementAndGet();
                      return new Object();
                    }));

    assertSame(streamFailure, thrown);
    assertEquals(1, streamCalls.get());
    assertEquals(0, builderCalls.get());
  }

  @Test
  public void testDeviceMatrixBuilderPassesAcquiredStreamToConstructor() throws Throwable {
    MemorySegment stream = MemorySegment.ofArray(new byte[1]);
    Object builder = new Object();
    AtomicInteger streamCalls = new AtomicInteger();
    AtomicInteger builderCalls = new AtomicInteger();
    AtomicReference<MemorySegment> observedStream = new AtomicReference<>();

    Object created =
        createDeviceMatrixBuilder(
            () -> {
              streamCalls.incrementAndGet();
              return stream;
            },
            actualStream -> {
              builderCalls.incrementAndGet();
              observedStream.set(actualStream);
              return builder;
            });

    assertSame(builder, created);
    assertSame(stream, observedStream.get());
    assertEquals(1, streamCalls.get());
    assertEquals(1, builderCalls.get());
  }

  static void checkCuVSVersionMatching(String mavenVersionString, int major, int minor, int patch)
      throws ProviderInitializationException {
    try {
      var cls = Class.forName("com.nvidia.cuvs.spi.JDKProvider");
      var ctr =
          MethodHandles.lookup()
              .findStatic(
                  cls,
                  "checkCuVSVersionMatching",
                  MethodType.methodType(
                      void.class, String.class, short.class, short.class, short.class));
      ctr.invoke(mavenVersionString, (short) major, (short) minor, (short) patch);
    } catch (ProviderInitializationException e) {
      throw e;
    } catch (Throwable e) {
      throw new AssertionError(e);
    }
  }

  static Object createDeviceMatrixBuilder(
      Supplier<MemorySegment> streamSupplier, Function<MemorySegment, Object> builderFactory)
      throws Throwable {
    var cls = Class.forName("com.nvidia.cuvs.spi.JDKProvider");
    var method =
        MethodHandles.lookup()
            .findStatic(
                cls,
                "createDeviceMatrixBuilder",
                MethodType.methodType(Object.class, Supplier.class, Function.class));
    return method.invoke(streamSupplier, builderFactory);
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal.common;

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;
import static org.hamcrest.CoreMatchers.equalTo;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.Assert.assertThrows;

import com.nvidia.cuvs.CuVSTestCase;
import com.nvidia.cuvs.LibraryException;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.Proxy;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class UtilIT extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(UtilIT.class);

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxSupportedArch());
  }

  @Test
  public void testGetLastErrorText() throws Throwable {
    var cls = Class.forName("com.nvidia.cuvs.internal.common.Util");
    var lookup = MethodHandles.lookup();
    var mt = MethodType.methodType(String.class);
    var mh = lookup.findStatic(cls, "getLastErrorText", mt);

    // first, ensures that accessing the error text when there is none does not crash!
    String errorText = (String) mh.invoke();
    // second, ensures that the default test is returned
    assertThat(errorText, equalTo("no last error text"));
  }

  @Test
  public void testCudaCallFailureRemainsLibraryException() throws Throwable {
    var cls = Class.forName("com.nvidia.cuvs.internal.common.Util");
    var cudaCallClass = Class.forName("com.nvidia.cuvs.internal.common.Util$CudaCall");
    var call =
        Proxy.newProxyInstance(
            cudaCallClass.getClassLoader(),
            new Class<?>[] {cudaCallClass},
            (proxy, method, args) -> 1);
    var methodType = MethodType.methodType(void.class, String.class, cudaCallClass);
    var invokeCuda = MethodHandles.lookup().findStatic(cls, "invokeCuda", methodType);

    LibraryException failure =
        assertThrows(LibraryException.class, () -> invokeCuda.invoke("testCudaCall", call));

    assertThat(failure.getMessage(), equalTo("testCudaCall returned 1"));
  }
}

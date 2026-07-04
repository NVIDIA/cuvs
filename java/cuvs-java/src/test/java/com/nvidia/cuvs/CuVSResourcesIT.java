/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;
import static org.junit.Assert.*;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CuVSResourcesIT extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(CuVSResourcesIT.class);

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxAmd64());
  }

  @Test
  public void testConcurrentAccessViaCheckedCuVSResourcesIsForbidden() throws Throwable {
    var expectedError =
        "This resource is already accessed by thread [" + Thread.currentThread().threadId() + "]";
    try (var resources = CheckedCuVSResources.create();
        var executor = Executors.newFixedThreadPool(1)) {

      try (var access1 = resources.access()) {
        log.debug(
            "Outer access to resource {} from {}",
            access1.handle(),
            Thread.currentThread().threadId());

        var exception =
            assertThrows(
                ExecutionException.class,
                () -> {
                  var future =
                      executor.submit(
                          () -> {
                            try (var access2 = resources.access()) {
                              log.debug(
                                  "Nested access to resource {} from {}",
                                  access2.handle(),
                                  Thread.currentThread().threadId());
                              log.debug("Nested access finished");
                            }
                          });
                  future.get();
                });
        assertEquals(IllegalStateException.class, exception.getCause().getClass());
        assertTrue(exception.getCause().getMessage().startsWith(expectedError));
        log.debug("Outer access finished");
      }
    }
  }

  @Test
  public void testSetWorkspacePoolRejectsNonPositiveSize() throws Throwable {
    try (var resources = CuVSResources.create()) {
      assertThrows(IllegalArgumentException.class, () -> resources.setWorkspacePool(0));
      assertThrows(IllegalArgumentException.class, () -> resources.setWorkspacePool(-1));
    }
  }

  @Test
  public void testSequentialAccessViaCheckedCuVSResourcesIsAllowed() throws Throwable {
    try (var resources = CheckedCuVSResources.create();
        var executor = Executors.newFixedThreadPool(1)) {

      try (var access1 = resources.access()) {
        log.debug(
            "Access 1 to resource {} from {}", access1.handle(), Thread.currentThread().threadId());
      }

      var future =
          executor.submit(
              () -> {
                try (var access2 = resources.access()) {
                  log.debug(
                      "Access 2 to resource {} from {}",
                      access2.handle(),
                      Thread.currentThread().threadId());
                }
              });
      future.get();
    }
  }
}

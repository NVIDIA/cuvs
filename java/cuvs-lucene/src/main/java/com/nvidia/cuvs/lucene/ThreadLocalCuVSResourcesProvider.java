/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CuVSResources;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Provides a mechanism to create ThreadLocal based CuVSResource instances.
 *
 * @since 26.02
 */
public class ThreadLocalCuVSResourcesProvider {

  private static final Logger log =
      Logger.getLogger(ThreadLocalCuVSResourcesProvider.class.getName());
  private static final ThreadLocal<CuVSResources> cuVSResources;

  static {
    cuVSResources = ThreadLocal.withInitial(() -> cuVSResourcesOrNull());
  }

  /**
   * Gets an instance of CuVSResources for the accessing thread.
   *
   * @return an instance of CuVSResources
   */
  public static CuVSResources getCuVSResourcesInstance() {
    return cuVSResources.get();
  }

  /**
   * Sets the instance of CuVSResources
   *
   * @param resources the instance of CuVSResources to set
   */
  public static void setCuVSResourcesInstance(CuVSResources resources) {
    cuVSResources.set(resources);
  }

  private static CuVSResources cuVSResourcesOrNull() {
    try {
      return CuVSResources.create();
    } catch (UnsupportedOperationException uoe) {
      log.log(
          Level.WARNING,
          "cuVS is not supported on this platform or java version: " + uoe.getMessage());
    } catch (Throwable t) {
      if (t instanceof ExceptionInInitializerError ex) {
        t = ex.getCause();
      }
      log.log(Level.WARNING, "Exception occurred during creation of cuVS resources. " + t);
    }
    return null;
  }

  /**
   * Attempts to close the thread's {@link CuVSResources} instance.
   */
  public static void closeCuVSResourcesInstance() {
    CuVSResources r = cuVSResources.get();
    if (r != null) {
      r.close();
    }
    cuVSResources.remove();
  }

  /**
   * Checks if cuVS is supported and throws {@link UnsupportedOperationException} otherwise.
   *
   * @throws UnsupportedOperationException
   */
  public static void assertIsSupported() throws UnsupportedOperationException {
    if (cuVSResources.get() == null) {
      throw new UnsupportedOperationException("cuVS is not supported");
    }
  }

  /**
   * Checks if cuVS is supported.
   *
   * @return true if cuVS is supported else false
   */
  public static boolean isSupported() {
    return cuVSResources.get() != null;
  }
}

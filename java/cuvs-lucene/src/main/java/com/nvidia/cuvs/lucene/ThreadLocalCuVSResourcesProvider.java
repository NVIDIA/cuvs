/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

  /** System property controlling the workspace pool size per resources handle (in bytes). */
  public static final String WORKSPACE_POOL_SIZE_PROPERTY = "com.nvidia.cuvs.workspacePoolSize";

  private static final long RMM_ALIGNMENT_BYTES = 256;

  private static CuVSResources cuVSResourcesOrNull() {
    CuVSResources resources = null;
    try {
      // Resolve configuration before allocating resources so malformed input cannot leak a newly
      // created native handle and pinned host buffer.
      long poolBytes =
          resolveWorkspacePoolBytes(System.getProperty(WORKSPACE_POOL_SIZE_PROPERTY));
      resources = CuVSResources.create();
      if (poolBytes > 0) {
        resources.setWorkspacePool(poolBytes);
      }
      return resources;
    } catch (UnsupportedOperationException uoe) {
      closeAfterFailedInitialization(resources, uoe);
      log.log(
          Level.WARNING,
          "cuVS is not supported on this platform or java version: " + uoe.getMessage());
    } catch (Throwable t) {
      Throwable failure = t;
      if (t instanceof ExceptionInInitializerError ex && ex.getCause() != null) {
        failure = ex.getCause();
      }
      closeAfterFailedInitialization(resources, failure);
      log.log(Level.WARNING, "Exception occurred during creation of cuVS resources. " + failure);
    }
    return null;
  }

  /**
   * Resolves a raw workspace-pool property value to a 256-byte-aligned size. Zero or an absent
   * value disables the per-resources pool. Invalid, negative, or unalignable values warn and also
   * disable it.
   */
  static long resolveWorkspacePoolBytes(String raw) {
    if (raw == null) return 0;

    final long requestedBytes;
    try {
      requestedBytes = Long.parseLong(raw.trim());
    } catch (NumberFormatException invalid) {
      warnInvalidWorkspacePoolSize(raw);
      return 0;
    }

    if (requestedBytes == 0) return 0;
    if (requestedBytes < 0
        || requestedBytes > Long.MAX_VALUE - (RMM_ALIGNMENT_BYTES - 1)) {
      warnInvalidWorkspacePoolSize(raw);
      return 0;
    }

    return (requestedBytes + (RMM_ALIGNMENT_BYTES - 1)) & ~(RMM_ALIGNMENT_BYTES - 1);
  }

  private static void warnInvalidWorkspacePoolSize(String raw) {
    log.warning(
        "Invalid "
            + WORKSPACE_POOL_SIZE_PROPERTY
            + " value \""
            + raw
            + "\"; expected a non-negative byte count that can be aligned to 256 bytes. "
            + "Continuing without a workspace pool.");
  }

  private static void closeAfterFailedInitialization(CuVSResources resources, Throwable failure) {
    if (resources == null) return;
    try {
      resources.close();
    } catch (Throwable closeFailure) {
      failure.addSuppressed(closeFailure);
    }
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

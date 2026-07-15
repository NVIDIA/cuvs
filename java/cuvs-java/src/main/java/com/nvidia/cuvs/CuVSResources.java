/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import com.nvidia.cuvs.spi.CuVSProvider;
import java.nio.file.Path;
import java.time.Duration;

/**
 * Used for allocating resources for cuVS
 *
 * @since 25.02
 */
public interface CuVSResources extends AutoCloseable {

  /**
   * Provide scoped access to the native resources object.
   */
  interface ScopedAccess extends AutoCloseable {
    /**
     * Gets the opaque CuVSResources handle, to be used whenever we need to pass a cuvsResources_t parameter
     *
     * @return the CuVSResources handle
     */
    long handle();

    @Override
    void close();
  }

  /**
   * Gets scoped access to the native resources object.
   * The native resource object is not thread safe: only a single thread at every time should access
   * concurrently the same native resources. Calling this method from multiple thread is OK, but the
   * returned {@link ScopedAccess} object must be closed before calling {@code access()} again from a
   * different thread.
   */
  ScopedAccess access();

  /**
   * Get the logical id of the device associated with this resources object.
   * Information about the device id is immutable, so it is safe to expose it without getting {@link ScopedAccess}
   * to the enclosing resources.
   */
  int deviceId();

  /**
   * Closes this CuVSResources object and releases any resources associated with it.
   */
  @Override
  void close();

  /**
   * The temporary directory to use for intermediate operations.
   * Defaults to {@systemProperty java.io.tmpdir}.
   */
  Path tempDirectory();

  /**
   * Creates a new resources.
   * Equivalent to
   * <pre>{@code
   *   create(CuVSProvider.tempDirectory())
   * }</pre>
   */
  static CuVSResources create() throws Throwable {
    return create(CuVSProvider.tempDirectory());
  }

  /**
   * Creates a new resources.
   *
   * @param tempDirectory the temporary directory to use for intermediate operations
   * @throws UnsupportedOperationException if the provider does not cuvs
   * @throws LibraryException if the native library cannot be loaded
   */
  static CuVSResources create(Path tempDirectory) throws Throwable {
    return CuVSProvider.provider().newCuVSResources(tempDirectory);
  }

  /**
   * Creates a new resources whose memory allocations are tracked and written as
   * CSV samples from a background thread.
   * <p>
   * The returned handle wraps all reachable memory resources (host, pinned,
   * managed, device, workspace, large_workspace) with allocation-tracking
   * adaptors and replaces the global host and device memory resources for the
   * lifetime of the handle. It is otherwise indistinguishable from a handle
   * created by {@link #create(Path)} and can be used wherever a
   * {@link CuVSResources} is accepted. The CSV reporter is stopped and the
   * global memory resources are restored when the handle is closed.
   *
   * @param tempDirectory                   the temporary directory to use for
   *                                        intermediate operations
   * @param memoryTrackingCsvPath           path to the output CSV file
   *                                        (created/truncated)
   * @param memoryTrackingSampleInterval    minimum interval between successive
   *                                        CSV samples
   * @throws UnsupportedOperationException if the provider does not support cuvs
   * @throws LibraryException if the native library cannot be loaded
   */
  static CuVSResources create(
      Path tempDirectory,
      Path memoryTrackingCsvPath,
      Duration memoryTrackingSampleInterval) throws Throwable {
    return CuVSProvider.provider()
        .newCuVSResources(
            tempDirectory, memoryTrackingCsvPath, memoryTrackingSampleInterval);
  }
}

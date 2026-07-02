/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import com.nvidia.cuvs.spi.CuVSProvider;
import java.nio.file.Path;

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
   * Configure the temporary workspace on this resources object as an uncapped pool backed by the
   * current device memory resource. After the initial reservation is allocated on first use,
   * subsequent calls to {@code cuvsRMMAlloc} / {@code cuvsRMMFree} on this handle hit the pool
   * cache rather than calling {@code cudaMallocAsync} / {@code cudaFreeAsync}, reducing CUDA
   * context lock contention under concurrent query threads. The pool grows without shrinking:
   * freed allocations are returned to the pool rather than to the device, so the pool's
   * high-water mark only increases until the resources object is closed.
   *
   * <p>The pool is per-resources-handle (i.e. per query thread when resources are thread-local),
   * so there is no cross-thread pool mutex contention. Call this once after creating the resources
   * object; calling it again replaces the pool.
   *
   * @param initialSizeBytes initial pool reservation in bytes; size {@code initialSizeBytes} to
   *                         cover the steady-state working set to avoid growth after warmup
   */
  void setWorkspacePool(long initialSizeBytes);

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
}

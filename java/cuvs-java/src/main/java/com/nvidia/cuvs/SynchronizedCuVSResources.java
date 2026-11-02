/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import java.nio.file.Path;
import java.util.concurrent.locks.ReentrantLock;

/**
 * A decorator for CuVSResources that guarantees synchronized (blocking) access to the wrapped CuVSResource
 */
public class SynchronizedCuVSResources implements CuVSResources {

  private final CuVSResources inner;
  private final ReentrantLock lock;

  private SynchronizedCuVSResources(CuVSResources inner) {
    this.inner = inner;
    this.lock = new ReentrantLock();
  }

  static CuVSResources create() throws Throwable {
    return new SynchronizedCuVSResources(CuVSResources.create());
  }

  @Override
  public ScopedAccess access() {
    lock.lock();
    return new DelegatingScopedAccess(inner.access(), lock::unlock);
  }

  @Override
  public int deviceId() {
    return inner.deviceId();
  }

  @Override
  public void close() {
    inner.close();
  }

  @Override
  public void setWorkspacePool(long sizeBytes) {
    lock.lock();
    try {
      inner.setWorkspacePool(sizeBytes);
    } finally {
      lock.unlock();
    }
  }

  @Override
  public Path tempDirectory() {
    return inner.tempDirectory();
  }
}

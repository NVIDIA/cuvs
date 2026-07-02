/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicLong;

public class CheckedCuVSResources implements CuVSResources {

  private volatile boolean destroyed;
  private final AtomicLong currentThreadId = new AtomicLong(0);

  private final CuVSResources inner;

  private CheckedCuVSResources(CuVSResources inner) {
    this.inner = inner;
  }

  static CuVSResources create() throws Throwable {
    return new CheckedCuVSResources(CuVSResources.create());
  }

  private void checkNotDestroyed() {
    if (destroyed) {
      throw new IllegalStateException("Already destroyed");
    }
  }

  @Override
  public ScopedAccess access() {
    checkNotDestroyed();
    var currentThreadId = Thread.currentThread().threadId();
    var previousThreadId = this.currentThreadId.compareAndExchange(0, currentThreadId);
    if (previousThreadId != 0 && previousThreadId != currentThreadId) {
      throw new IllegalStateException(
          "This resource is already accessed by thread ["
              + previousThreadId
              + "]. Current thread id: ["
              + currentThreadId
              + "]");
    }
    return new DelegatingScopedAccess(
        inner.access(), () -> CheckedCuVSResources.this.currentThreadId.set(0));
  }

  @Override
  public int deviceId() {
    return inner.deviceId();
  }

  @Override
  public void close() {
    destroyed = true;
    inner.close();
  }

  @Override
  public void setWorkspacePool(long sizeBytes) {
    inner.setWorkspacePool(sizeBytes);
  }

  @Override
  public Path tempDirectory() {
    return inner.tempDirectory();
  }
}

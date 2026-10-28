/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal.common;

import static com.nvidia.cuvs.internal.panama.headers_h.omp_get_max_threads;
import static com.nvidia.cuvs.internal.panama.headers_h.omp_set_num_threads;

/**
 * Sets the OpenMP thread count for the native work done by the current thread, and restores the
 * previous value on {@link #close()}.
 *
 * <p>OpenMP's {@code nthreads-var} is per thread: {@code omp_set_num_threads} only affects parallel
 * regions started by the calling thread, so concurrent builds on other threads keep their own
 * values. What must not happen is leaving this thread's value changed after the call returns, which
 * is why the previous value is restored rather than reset to a fixed number.
 */
public final class OmpThreadScope implements AutoCloseable {
  private final int previous;

  private OmpThreadScope(int previous) {
    this.previous = previous;
  }

  /**
   * Applies {@code numThreads} for the current thread until the returned scope is closed. A value
   * of zero or less leaves the calling thread's current setting untouched.
   */
  public static OmpThreadScope of(int numThreads) {
    if (numThreads <= 0) {
      return new OmpThreadScope(0);
    }
    int previous = omp_get_max_threads();
    omp_set_num_threads(numThreads);
    return new OmpThreadScope(previous);
  }

  @Override
  public void close() {
    if (previous > 0) {
      omp_set_num_threads(previous);
    }
  }
}

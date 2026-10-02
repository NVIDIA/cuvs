/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import org.apache.lucene.util.Version;

/**
 * Detects a cuvs-lucene artifact running against a Lucene release other than the one it was built
 * for.
 *
 * <p>Lucene changes its codec APIs between minor releases, and some of those changes do not show
 * up as linkage errors: a method whose signature changed simply stops overriding its Lucene
 * counterpart, and Lucene silently runs its own implementation instead. Checking the version turns
 * that into a clear error.
 *
 * <p>The check must only run where cuvs-lucene is actually used, never while Lucene's service
 * loader instantiates the registered codecs and formats: an exception there would break every codec
 * and format lookup in the JVM, including Lucene's own.
 */
final class LuceneVersionGuard {

  private LuceneVersionGuard() {}

  /**
   * Throws unless the Lucene on the classpath is the release this artifact is built for.
   *
   * @throws IllegalStateException if another Lucene release is on the classpath
   */
  static void ensureCompatible() {
    String message = mismatchMessage(Version.LATEST);
    if (message != null) {
      throw new IllegalStateException(message);
    }
  }

  /**
   * Returns why this artifact cannot run on the given Lucene release, or {@code null} if it can.
   *
   * @param running the Lucene release in use
   * @return the error message, or null
   */
  static String mismatchMessage(Version running) {
    int major = LuceneCompat.LUCENE_MAJOR;
    int minor = LuceneCompat.LUCENE_MINOR;
    if (running.major == major && running.minor == minor) {
      return null;
    }
    // Only this artifact's own release is known here, so name the one to use without claiming it
    // exists: this cuvs-lucene release may not support that Lucene release at all.
    return String.format(
        "cuvs-lucene-%d.%d is built for Lucene %d.%d.x, but Lucene %s is in use. Use the"
            + " cuvs-lucene-%d.%d artifact of this cuvs-lucene release; if there is none, this"
            + " cuvs-lucene release does not support Lucene %d.%d.",
        major,
        minor,
        major,
        minor,
        running,
        running.major,
        running.minor,
        running.major,
        running.minor);
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

/** Shared range checks for public Lucene parameters. */
final class ParameterValidation {
  private ParameterValidation() {}

  static void checkRange(String name, long value, long min, long max) {
    if (value < min || value > max) {
      throw new IllegalArgumentException(
          name + " not in valid range. Valid range: [" + min + ", " + max + "], but was " + value);
    }
  }
}

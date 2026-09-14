/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.apache.lucene.tests.util.LuceneTestCase;

public class TestExternalFloat32Dataset extends LuceneTestCase {

  public void testRejectsWritableMemory() {
    try (Arena arena = Arena.ofShared()) {
      MemorySegment writable = arena.allocate(2L * 3 * Float.BYTES, Float.BYTES);

      expectThrows(
          IllegalArgumentException.class,
          () -> ExternalFloat32Dataset.fromMemorySegment(writable, 2, 3));
    }
  }

  public void testRejectsMisalignedMemory() {
    try (Arena arena = Arena.ofShared()) {
      MemorySegment misaligned =
          arena
              .allocate(2L * 3 * Float.BYTES + 1, Float.BYTES)
              .asSlice(1, 2L * 3 * Float.BYTES)
              .asReadOnly();

      expectThrows(
          IllegalArgumentException.class,
          () -> ExternalFloat32Dataset.fromMemorySegment(misaligned, 2, 3));
    }
  }
}

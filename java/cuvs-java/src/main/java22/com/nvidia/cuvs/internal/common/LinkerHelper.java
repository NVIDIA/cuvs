/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal.common;

import java.lang.foreign.AddressLayout;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.ValueLayout;

/**
 * Utility methods for calling into the native linker.
 */
public class LinkerHelper {

  private static final Linker LINKER = Linker.nativeLinker();

  public static final ValueLayout.OfByte C_CHAR =
      (ValueLayout.OfByte) LINKER.canonicalLayouts().get("char");

  public static final ValueLayout.OfInt C_INT =
      (ValueLayout.OfInt) LINKER.canonicalLayouts().get("int");

  public static final ValueLayout.OfLong C_LONG =
      (ValueLayout.OfLong) LINKER.canonicalLayouts().get("long");

  public static final ValueLayout.OfShort C_SHORT =
      (ValueLayout.OfShort) LINKER.canonicalLayouts().get("short");

  public static final ValueLayout.OfFloat C_FLOAT =
      (ValueLayout.OfFloat) LINKER.canonicalLayouts().get("float");

  public static final AddressLayout C_POINTER =
      ((AddressLayout) LINKER.canonicalLayouts().get("void*"))
          .withTargetLayout(MemoryLayout.sequenceLayout(Long.MAX_VALUE, C_CHAR));

  public static final long C_INT_BYTE_SIZE = LinkerHelper.C_INT.byteSize();

  public static final long C_FLOAT_BYTE_SIZE = LinkerHelper.C_FLOAT.byteSize();

  public static final long C_LONG_BYTE_SIZE = LinkerHelper.C_LONG.byteSize();

  private LinkerHelper() {}
}

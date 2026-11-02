/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal;

import java.util.BitSet;

/**
 * Packs prefilter {@link BitSet}s into the bit layout cuVS reads for its BITSET and BITMAP filters.
 *
 * <p>Filter {@code i} occupies bits {@code [i * numRows, (i + 1) * numRows)} of one contiguous
 * little-endian bit array, with no padding between filters, so in a bitmap every filter after the
 * first can start in the middle of a word. Bit {@code b} is bit {@code b % 64} of word {@code b /
 * 64}. cuVS reads the array on the GPU as little-endian 32-bit words, so copy the words returned here
 * to it little-endian: their bytes are then those of the 32-bit words, whatever the host's byte
 * order.
 *
 * <p>A {@code null} filter accepts every row, while a filter with no bits set rejects every row.
 * Bits at or beyond {@code numRows} are ignored.
 */
public final class BitsetPacker {

  private BitsetPacker() {}

  /** Returns the number of 32-bit words cuVS reads for {@code numBits} bits. */
  public static long words32(long numBits) {
    return (numBits + 31) / 32;
  }

  /**
   * Packs {@code filters}, {@code numRows} bits each, into 64-bit words.
   *
   * @param filters the filters to pack, in order; a {@code null} entry accepts every row
   * @param numRows the number of bits each filter contributes
   * @return {@code ceil(filters.length * numRows / 64)} words; the bits past the last filter are 0
   */
  public static long[] pack(BitSet[] filters, long numRows) {
    if (numRows < 0) {
      throw new IllegalArgumentException("numRows must not be negative, got " + numRows);
    }
    long totalBits = Math.multiplyExact(numRows, (long) filters.length);
    long numLongs = (totalBits + 63) / 64;
    if (numLongs > Integer.MAX_VALUE - 8) {
      throw new IllegalArgumentException(
          filters.length + " filters of " + numRows + " rows are too large to pack");
    }
    long[] packed = new long[(int) numLongs];
    for (int i = 0; i < filters.length; i++) {
      long base = i * numRows;
      if (filters[i] == null) {
        setRange(packed, base, base + numRows);
      } else {
        orBits(packed, base, filters[i], numRows);
      }
    }
    return packed;
  }

  /** ORs the first {@code numRows} bits of {@code filter} into {@code packed} at bit {@code base}. */
  private static void orBits(long[] packed, long base, BitSet filter, long numRows) {
    // toLongArray() stops at the last set bit, which is fine here: the array is zero-initialized.
    long[] src = filter.toLongArray();
    int srcLongs = (int) Math.min(src.length, (numRows + 63) / 64);
    int shift = (int) (base & 63);
    int dst = (int) (base >>> 6);
    for (int j = 0; j < srcLongs; j++) {
      long word = src[j];
      long remaining = numRows - 64L * j;
      if (remaining < 64) {
        word &= (1L << remaining) - 1;
      }
      if (word == 0) {
        continue;
      }
      packed[dst + j] |= word << shift;
      if (shift != 0 && dst + j + 1 < packed.length) {
        packed[dst + j + 1] |= word >>> (64 - shift);
      }
    }
  }

  /** Sets bits {@code [from, to)} of {@code packed}. */
  private static void setRange(long[] packed, long from, long to) {
    if (from >= to) {
      return;
    }
    int first = (int) (from >>> 6);
    int last = (int) ((to - 1) >>> 6);
    long firstMask = -1L << (from & 63);
    long lastMask = -1L >>> (63 - ((to - 1) & 63));
    if (first == last) {
      packed[first] |= firstMask & lastMask;
      return;
    }
    packed[first] |= firstMask;
    for (int i = first + 1; i < last; i++) {
      packed[i] = -1L;
    }
    packed[last] |= lastMask;
  }
}

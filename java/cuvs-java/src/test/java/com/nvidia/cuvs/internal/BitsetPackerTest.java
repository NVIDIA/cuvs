/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.Random;
import org.junit.Test;

public class BitsetPackerTest {

  /**
   * Row counts on both sides of the 32- and 64-bit word boundaries. 195 is the dataset size of the
   * prefilter leak in #1048: its last word holds only rows 192 to 194.
   */
  private static final long[] ROW_COUNTS = {0, 1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 192, 195};

  @Test
  public void packsEveryKindOfFilterAtEveryOffset() {
    Random random = new Random(42);
    for (long numRows : ROW_COUNTS) {
      List<BitSet> kinds = filterKinds(random, (int) numRows);
      // Every kind alone, and every kind after each other kind, so that each one also starts in
      // the middle of a word.
      for (BitSet kind : kinds) {
        assertPacked(new BitSet[] {kind}, numRows);
        for (BitSet first : kinds) {
          assertPacked(new BitSet[] {first, kind, first}, numRows);
        }
      }
    }
  }

  @Test
  public void emptyFilterRejectsEveryRow() {
    assertArrayEquals(new long[4], BitsetPacker.pack(new BitSet[] {new BitSet()}, 195));
  }

  @Test
  public void nullFilterAcceptsEveryRow() {
    long[] packed = BitsetPacker.pack(new BitSet[] {null}, 195);
    assertArrayEquals(new long[] {-1L, -1L, -1L, 0b111L}, packed);
  }

  @Test
  public void bitsBeyondNumRowsAreIgnored() {
    BitSet filter = new BitSet();
    filter.set(0, 200);
    BitSet second = new BitSet();
    second.set(2);
    // The first filter's bits at 3 and beyond must not spill into the second one.
    assertArrayEquals(new long[] {0b100111L}, BitsetPacker.pack(new BitSet[] {filter, second}, 3));
  }

  @Test
  public void noFiltersPackToNothing() {
    assertEquals(0, BitsetPacker.pack(new BitSet[0], 100).length);
  }

  @Test
  public void rejectsNegativeRowCount() {
    assertThrows(IllegalArgumentException.class, () -> BitsetPacker.pack(new BitSet[] {null}, -1));
  }

  @Test
  public void rejectsFiltersTooLargeToPack() {
    // More bits than a long[] can hold, without allocating anything.
    assertThrows(
        IllegalArgumentException.class, () -> BitsetPacker.pack(new BitSet[1 << 10], 1L << 40));
  }

  @Test
  public void countsWords32() {
    assertEquals(0, BitsetPacker.words32(0));
    assertEquals(1, BitsetPacker.words32(1));
    assertEquals(1, BitsetPacker.words32(32));
    assertEquals(2, BitsetPacker.words32(33));
    assertEquals(7, BitsetPacker.words32(195));
  }

  /** null, empty, full, random, only the last row, and bits set past numRows. */
  private static List<BitSet> filterKinds(Random random, int numRows) {
    List<BitSet> kinds = new ArrayList<>();
    kinds.add(null);
    kinds.add(new BitSet());
    BitSet full = new BitSet();
    full.set(0, numRows);
    kinds.add(full);
    BitSet randomBits = new BitSet();
    for (int i = 0; i < numRows; i++) {
      randomBits.set(i, random.nextBoolean());
    }
    kinds.add(randomBits);
    if (numRows > 0) {
      BitSet last = new BitSet();
      last.set(numRows - 1);
      kinds.add(last);
    }
    BitSet overflowing = (BitSet) randomBits.clone();
    overflowing.set(numRows, numRows + 70);
    kinds.add(overflowing);
    return kinds;
  }

  /** Compares the packed words with the bits each filter should contribute, one bit at a time. */
  private static void assertPacked(BitSet[] filters, long numRows) {
    long[] packed = BitsetPacker.pack(filters, numRows);
    long totalBits = numRows * filters.length;
    assertEquals((totalBits + 63) / 64, packed.length);
    for (long bit = 0; bit < 64L * packed.length; bit++) {
      boolean expected = false;
      if (bit < totalBits) {
        BitSet filter = filters[(int) (bit / numRows)];
        expected = filter == null || filter.get((int) (bit % numRows));
      }
      boolean actual = (packed[(int) (bit >>> 6)] & (1L << (bit & 63))) != 0;
      assertEquals("bit " + bit + " of " + filters.length + " x " + numRows, expected, actual);
    }
  }
}

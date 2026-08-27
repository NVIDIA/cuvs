/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.generateDataset;
import static com.nvidia.cuvs.lucene.TestUtils.writeFbin;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;
import java.util.UUID;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/**
 * Correctness coverage for {@link FbinVectorSource}: header parsing, whole-file vs. windowed
 * (sliced) reads, and the forward-only/single-consumer contract it shares with {@link
 * VectorSource}. Does not require GPU support -- this is pure file I/O.
 */
public class TestFbinVectorSource extends LuceneTestCase {

  private Path fbinPath;

  @Before
  public void beforeTest() {
    fbinPath = Paths.get(UUID.randomUUID() + ".fbin");
  }

  @After
  public void afterTest() {
    if (fbinPath != null) {
      new File(fbinPath.toString()).delete();
    }
  }

  @Test
  public void testWholeFileReadMatchesDataset() throws Exception {
    Random random = new Random(1);
    int numVectors = 250;
    int dimension = 17;
    float[][] dataset = generateDataset(random, numVectors, dimension);
    writeFbin(fbinPath, dataset);

    // A small chunk size forces multiple prefetch chunks so this also exercises the chunk
    // boundary/advance() path, not just a single-chunk read.
    try (FbinVectorSource source = new FbinVectorSource(fbinPath, /* chunkSizeMB= */ 1)) {
      assertEquals(dimension, source.dimensions());
      assertEquals(numVectors, source.size());

      float[] scratch = new float[dimension];
      for (int i = 0; i < numVectors; i++) {
        source.get(i, scratch);
        assertArrayEquals("vector " + i + " mismatch", dataset[i], scratch, 0f);
      }
    }
  }

  @Test
  public void testWindowedReadServesOnlyItsSlice() throws Exception {
    Random random = new Random(2);
    int numVectors = 100;
    int dimension = 8;
    int sliceStart = 30;
    int sliceSize = 25;
    float[][] dataset = generateDataset(random, numVectors, dimension);
    writeFbin(fbinPath, dataset);

    try (FbinVectorSource source = new FbinVectorSource(fbinPath, sliceStart, sliceSize, 1)) {
      assertEquals(dimension, source.dimensions());
      assertEquals(sliceSize, source.size());

      for (int i = 0; i < sliceSize; i++) {
        float[] got = source.get(i);
        assertArrayEquals(
            "relative index " + i + " should map to absolute " + (sliceStart + i),
            dataset[sliceStart + i],
            got,
            0f);
      }
    }
  }

  @Test
  public void testOutOfOrderAccessIsRejected() throws Exception {
    Random random = new Random(3);
    float[][] dataset = generateDataset(random, 20, 4);
    writeFbin(fbinPath, dataset);

    try (FbinVectorSource source = new FbinVectorSource(fbinPath, 1)) {
      source.get(5);
      try {
        source.get(2); // going backwards must be rejected: forward-only contract
        fail("expected UnsupportedOperationException for out-of-order access");
      } catch (UnsupportedOperationException expected) {
        // expected
      }
    }
  }

  @Test
  public void testOutOfBoundsIndexIsRejected() throws Exception {
    Random random = new Random(4);
    float[][] dataset = generateDataset(random, 10, 4);
    writeFbin(fbinPath, dataset);

    try (FbinVectorSource source = new FbinVectorSource(fbinPath, 1)) {
      try {
        source.get(10); // window is [0, 10)
        fail("expected IndexOutOfBoundsException");
      } catch (IndexOutOfBoundsException expected) {
        // expected
      }
    }
  }
}

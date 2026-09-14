/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Random;

public class TestUtils {

  /** Writes {@code dataset} as an uncompressed {@code .fbin} file (little-endian float32 rows). */
  public static void writeFbin(Path path, float[][] dataset) throws IOException {
    int numVectors = dataset.length;
    int dimension = numVectors == 0 ? 0 : dataset[0].length;
    ByteBuffer buf =
        ByteBuffer.allocate(8 + numVectors * dimension * Float.BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
    buf.putInt(numVectors);
    buf.putInt(dimension);
    for (float[] vector : dataset) {
      for (float v : vector) {
        buf.putFloat(v);
      }
    }
    buf.flip();
    try (FileChannel ch =
        FileChannel.open(
            path,
            StandardOpenOption.CREATE,
            StandardOpenOption.WRITE,
            StandardOpenOption.TRUNCATE_EXISTING)) {
      while (buf.hasRemaining()) {
        ch.write(buf);
      }
    }
  }

  public static float[][] generateDataset(Random random, int size, int dimensions) {
    float[][] dataset = new float[size][dimensions];
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < dimensions; j++) {
        dataset[i][j] = random.nextFloat() * 100;
      }
    }
    return dataset;
  }

  public static float[] generateRandomVector(int dimensions, Random random) {
    float[] vector = new float[dimensions];
    for (int i = 0; i < dimensions; i++) {
      vector[i] = random.nextFloat() * 100;
    }
    return vector;
  }

  public static float[][] generateQueries(Random random, int dimensions, int numQueries) {
    // Generate random query vectors
    float[][] queries = new float[numQueries][dimensions];
    for (int i = 0; i < numQueries; i++) {
      for (int j = 0; j < dimensions; j++) {
        queries[i][j] = random.nextFloat() * 100;
      }
    }
    return queries;
  }
}

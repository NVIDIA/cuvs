/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressFileSystems;
import org.junit.Test;

@SuppressFileSystems("*")
public class TestMappedFbinDataset extends LuceneTestCase {

  @Test
  public void testMapsDenseFloatPayloadWithoutCopying() throws Exception {
    Path file = createTempDir().resolve("vectors.fbin");
    writeFbin(file, 2, 3, new float[] {1, 2, 3, 4, 5, 6});

    try (MappedFbinDataset mapped = MappedFbinDataset.map(file)) {
      assertEquals(2, mapped.rows());
      assertEquals(3, mapped.dimensions());
      float[] second = new float[3];
      mapped.dataset().matrix().getRow(1).toArray(second);
      assertArrayEquals(new float[] {4, 5, 6}, second, 0.0f);
    }
  }

  @Test
  public void testRejectsTrailingOrTruncatedPayload() throws Exception {
    Path trailing = createTempDir().resolve("trailing.fbin");
    writeFbin(trailing, 1, 2, new float[] {1, 2, 3});
    assertThrows(IOException.class, () -> MappedFbinDataset.map(trailing));

    Path truncated = createTempDir().resolve("truncated.fbin");
    writeFbin(truncated, 2, 2, new float[] {1, 2, 3});
    assertThrows(IOException.class, () -> MappedFbinDataset.map(truncated));
  }

  private static void writeFbin(Path path, int rows, int dimensions, float[] values)
      throws IOException {
    ByteBuffer data =
        ByteBuffer.allocate(2 * Integer.BYTES + values.length * Float.BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
    data.putInt(rows).putInt(dimensions);
    for (float value : values) {
      data.putFloat(value);
    }
    data.flip();
    try (FileChannel channel =
        FileChannel.open(path, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
      while (data.hasRemaining()) {
        channel.write(data);
      }
    }
    assertTrue(Files.isRegularFile(path));
  }
}

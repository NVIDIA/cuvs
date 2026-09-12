/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.apache.lucene.tests.util.LuceneTestCase;

public class TestFbinFileMetadata extends LuceneTestCase {

  public void testReadsShapeWithoutAStreamingSource() throws Exception {
    Path file = createTempDir().resolve("vectors.fbin");
    TestUtils.writeFbin(file, new float[][] {{1f, 2f}, {3f, 4f}, {5f, 6f}});

    FbinFileMetadata metadata = FbinFileMetadata.read(file);

    assertEquals(3, metadata.rows());
    assertEquals(2, metadata.dimensions());
    assertEquals(Files.size(file), metadata.fileBytes());
  }

  public void testRejectsTrailingBytes() throws Exception {
    Path file = createTempDir().resolve("vectors.fbin");
    TestUtils.writeFbin(file, new float[][] {{1f, 2f}});
    Files.write(file, new byte[] {1}, java.nio.file.StandardOpenOption.APPEND);

    IOException failure = expectThrows(IOException.class, () -> FbinFileMetadata.read(file));
    assertTrue(failure.getMessage().contains("does not match"));
  }
}

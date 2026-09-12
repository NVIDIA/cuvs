/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/** Synchronously reads and validates an FBIN header without prefetching any vector payload. */
record FbinFileMetadata(int rows, int dimensions, long fileBytes) {

  private static final int HEADER_BYTES = 2 * Integer.BYTES;

  static FbinFileMetadata read(Path path) throws IOException {
    try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
      ByteBuffer header = ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
      long position = 0L;
      while (header.hasRemaining()) {
        int read = channel.read(header, position);
        if (read < 0) {
          throw new EOFException("Unexpected EOF while reading FBIN header from " + path);
        }
        if (read == 0) {
          throw new IOException("Unable to make progress while reading FBIN header from " + path);
        }
        position += read;
      }
      header.flip();
      int rows = header.getInt();
      int dimensions = header.getInt();
      if (rows <= 0 || dimensions <= 0) {
        throw new IOException(
            "FBIN header must contain positive rows and dimensions: " + rows + " x " + dimensions);
      }
      final long expectedBytes;
      try {
        expectedBytes =
            Math.addExact(
                HEADER_BYTES,
                Math.multiplyExact(Math.multiplyExact((long) rows, dimensions), Float.BYTES));
      } catch (ArithmeticException overflow) {
        throw new IOException("FBIN header shape overflows its file length", overflow);
      }
      long actualBytes = channel.size();
      if (actualBytes != expectedBytes) {
        throw new IOException(
            "FBIN length "
                + actualBytes
                + " does not match header-derived length "
                + expectedBytes);
      }
      return new FbinFileMetadata(rows, dimensions, actualBytes);
    }
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.EOFException;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/** Owned read-only mapping of a structurally validated dense float32 FBIN payload. */
final class MappedFbinDataset implements AutoCloseable {

  private static final long HEADER_BYTES = 2L * Integer.BYTES;

  private final Arena arena;
  private final ExternalFloat32Dataset dataset;
  private boolean closed;

  static MappedFbinDataset map(Path path) throws IOException {
    int rows;
    int dimensions;
    long payloadBytes;
    Arena arena = Arena.ofShared();
    try {
      MemorySegment payload;
      try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
        ByteBuffer header = ByteBuffer.allocate((int) HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        readFully(channel, header, 0L);
        header.flip();
        rows = header.getInt();
        dimensions = header.getInt();
        if (rows <= 0 || dimensions <= 0) {
          throw new IOException(
              "FBIN header must contain positive rows and dimensions: "
                  + rows
                  + " x "
                  + dimensions);
        }
        payloadBytes = Math.multiplyExact(Math.multiplyExact((long) rows, dimensions), Float.BYTES);
        long expectedBytes = Math.addExact(HEADER_BYTES, payloadBytes);
        if (channel.size() != expectedBytes) {
          throw new IOException(
              "FBIN length "
                  + channel.size()
                  + " does not match header-derived length "
                  + expectedBytes);
        }
        payload = channel.map(FileChannel.MapMode.READ_ONLY, HEADER_BYTES, payloadBytes, arena);
      }
      ExternalFloat32Dataset dataset =
          ExternalFloat32Dataset.fromMemorySegment(payload, rows, dimensions);
      return new MappedFbinDataset(arena, dataset);
    } catch (IOException | RuntimeException | Error failure) {
      arena.close();
      throw failure;
    }
  }

  private static void readFully(FileChannel channel, ByteBuffer target, long position)
      throws IOException {
    long current = position;
    while (target.hasRemaining()) {
      int read = channel.read(target, current);
      if (read < 0) {
        throw new EOFException("Unexpected EOF while reading FBIN header");
      }
      if (read == 0) {
        throw new IOException("Unable to make progress while reading FBIN header");
      }
      current += read;
    }
  }

  private MappedFbinDataset(Arena arena, ExternalFloat32Dataset dataset) {
    this.arena = arena;
    this.dataset = dataset;
  }

  ExternalFloat32Dataset dataset() {
    ensureOpen();
    return dataset;
  }

  int rows() {
    return dataset.rows();
  }

  int dimensions() {
    return dataset.dimensions();
  }

  private void ensureOpen() {
    if (closed) {
      throw new IllegalStateException("Mapped FBIN dataset is closed");
    }
  }

  @Override
  public void close() {
    if (closed) {
      return;
    }
    closed = true;
    try {
      dataset.matrix().close();
    } finally {
      arena.close();
    }
  }
}

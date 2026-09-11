/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
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
    FbinFileMetadata metadata = FbinFileMetadata.read(path);
    return map(path, metadata, 0, metadata.rows());
  }

  static MappedFbinDataset map(Path path, int firstRow, int rowCount) throws IOException {
    FbinFileMetadata metadata = FbinFileMetadata.read(path);
    return map(path, metadata, firstRow, rowCount);
  }

  private static MappedFbinDataset map(
      Path path, FbinFileMetadata metadata, int firstRow, int rowCount) throws IOException {
    if (firstRow < 0 || rowCount <= 0) {
      throw new IllegalArgumentException("firstRow must be non-negative and rowCount positive");
    }
    if ((long) firstRow + rowCount > metadata.rows()) {
      throw new IllegalArgumentException(
          "Requested row range ["
              + firstRow
              + ", "
              + ((long) firstRow + rowCount)
              + ") exceeds FBIN row count "
              + metadata.rows());
    }

    long rowBytes = Math.multiplyExact((long) metadata.dimensions(), Float.BYTES);
    long payloadOffset = Math.addExact(HEADER_BYTES, Math.multiplyExact((long) firstRow, rowBytes));
    long payloadBytes = Math.multiplyExact((long) rowCount, rowBytes);
    Arena arena = Arena.ofShared();
    try {
      MemorySegment payload;
      try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
        if (channel.size() != metadata.fileBytes()) {
          throw new IOException(
              "FBIN length changed before mapping: expected "
                  + metadata.fileBytes()
                  + " but got "
                  + channel.size()
                  + " for "
                  + path);
        }
        payload = channel.map(FileChannel.MapMode.READ_ONLY, payloadOffset, payloadBytes, arena);
      }
      ExternalFloat32Dataset dataset =
          ExternalFloat32Dataset.fromMemorySegment(payload, rowCount, metadata.dimensions());
      return new MappedFbinDataset(arena, dataset);
    } catch (IOException | RuntimeException | Error failure) {
      arena.close();
      throw failure;
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

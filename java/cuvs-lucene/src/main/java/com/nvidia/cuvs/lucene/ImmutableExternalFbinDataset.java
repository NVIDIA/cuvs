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

/**
 * An owned read-only mapping whose native build bytes and persisted external reference are created
 * from the same registered FBIN range.
 *
 * <p>Instances are created by {@link ExternalFbinFileRegistry.Registration#map(int, int)}. This
 * coupled type prevents accidentally building a graph from one same-shaped native dataset while
 * persisting a reference to another. Closing it releases both the cuVS matrix wrapper and mapping;
 * callers must keep it open until the index writer has finished.
 */
final class ImmutableExternalFbinDataset implements AutoCloseable {

  private final ExternalFbinReference reference;
  private final Arena arena;
  private final ExternalFloat32Dataset dataset;
  private boolean closed;

  static ImmutableExternalFbinDataset map(ExternalFbinReference reference) throws IOException {
    Path sourcePath = ExternalFbinIO.validateAndResolve(reference);
    Arena arena = Arena.ofShared();
    try {
      MemorySegment payload;
      try (FileChannel channel = FileChannel.open(sourcePath, StandardOpenOption.READ)) {
        long actualLength = channel.size();
        if (actualLength != reference.fileLength()) {
          throw new IOException(
              "External FBIN length changed before mapping: expected "
                  + reference.fileLength()
                  + " but got "
                  + actualLength);
        }
        payload =
            channel.map(
                FileChannel.MapMode.READ_ONLY,
                reference.payloadOffset(),
                reference.payloadLength(),
                arena);
      }
      ExternalFloat32Dataset dataset =
          ExternalFloat32Dataset.fromMemorySegment(
              payload, reference.rows(), reference.dimensions());
      return new ImmutableExternalFbinDataset(reference, arena, dataset);
    } catch (IOException | RuntimeException | Error failure) {
      arena.close();
      throw failure;
    }
  }

  private ImmutableExternalFbinDataset(
      ExternalFbinReference reference, Arena arena, ExternalFloat32Dataset dataset) {
    this.reference = reference;
    this.arena = arena;
    this.dataset = dataset;
  }

  ExternalFloat32Dataset dataset() {
    ensureOpen();
    return dataset;
  }

  ExternalFbinReference reference() {
    ensureOpen();
    return reference;
  }

  int rows() {
    return reference.rows();
  }

  int dimensions() {
    return reference.dimensions();
  }

  private synchronized void ensureOpen() {
    if (closed) {
      throw new IllegalStateException("Immutable external FBIN dataset is closed");
    }
  }

  @Override
  public synchronized void close() {
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

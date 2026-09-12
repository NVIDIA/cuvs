/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Objects;
import java.util.function.LongConsumer;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.store.IndexInput;

final class ExternalFbinIO {

  private static final int BUFFER_BYTES = 8 << 20;

  private ExternalFbinIO() {}

  static Path validateAndResolve(ExternalFbinReference reference) throws IOException {
    Path path = ExternalFbinFileRegistry.resolve(reference);
    final ExternalFbinReference actual;
    try {
      actual =
          ExternalFbinReference.fromFile(
              path, reference.sha256Hex(), Math.toIntExact(reference.firstRow()), reference.rows());
    } catch (IllegalArgumentException | ArithmeticException incompatibleFile) {
      throw new IOException(
          "Registered external FBIN "
              + path
              + " is incompatible with persisted reference "
              + reference.contentId(),
          incompatibleFile);
    }
    if (!actual.equals(reference)) {
      throw new IOException(
          "Registered FBIN metadata does not match persisted reference for "
              + reference.contentId()
              + ": expected "
              + reference
              + " but found "
              + actual);
    }
    return path;
  }

  static long prefetch(ExternalFbinReference reference) throws IOException {
    return prefetch(reference, ignored -> {});
  }

  static long prefetch(ExternalFbinReference reference, LongConsumer progress) throws IOException {
    Path path = validateAndResolve(reference);
    return scan(path, reference.payloadOffset(), reference.payloadLength(), null, progress);
  }

  static long verifySha256(ExternalFbinReference reference) throws IOException {
    return verifySha256(reference, ignored -> {});
  }

  static long verifySha256(ExternalFbinReference reference, LongConsumer progress)
      throws IOException {
    Path path = validateAndResolve(reference);
    MessageDigest digest = newSha256();
    long scanned = scan(path, 0L, reference.fileLength(), digest, progress);
    byte[] actual = digest.digest();
    if (!MessageDigest.isEqual(reference.sha256(), actual)) {
      throw new IOException(
          "External FBIN SHA-256 mismatch for "
              + path
              + ": expected "
              + reference.sha256Hex()
              + " but got "
              + java.util.HexFormat.of().formatHex(actual));
    }
    return scanned;
  }

  static void verifySha256(IndexInput source, ExternalFbinReference reference) throws IOException {
    if (source.length() != reference.fileLength()) {
      throw new CorruptIndexException(
          "External FBIN length changed: expected "
              + reference.fileLength()
              + " but opened "
              + source.length(),
          source);
    }
    MessageDigest digest = newSha256();
    byte[] buffer = new byte[BUFFER_BYTES];
    // This is a non-owning clone. In particular, closing a multi-chunk MemorySegmentIndexInput
    // clone clears the segment array shared with its live owner.
    IndexInput clone = source.clone();
    clone.seek(0L);
    long remaining = clone.length();
    while (remaining > 0) {
      int length = (int) Math.min(buffer.length, remaining);
      clone.readBytes(buffer, 0, length);
      digest.update(buffer, 0, length);
      remaining -= length;
    }
    byte[] actual = digest.digest();
    if (!MessageDigest.isEqual(reference.sha256(), actual)) {
      throw new CorruptIndexException(
          "External FBIN SHA-256 mismatch: expected "
              + reference.sha256Hex()
              + " but got "
              + java.util.HexFormat.of().formatHex(actual),
          source);
    }
  }

  private static long scan(
      Path path, long offset, long length, MessageDigest optionalDigest, LongConsumer progress)
      throws IOException {
    Objects.requireNonNull(progress, "progress");
    ByteBuffer buffer = ByteBuffer.allocateDirect((int) Math.min(BUFFER_BYTES, length));
    long position = offset;
    long remaining = length;
    long scanned = 0L;
    try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
      while (remaining > 0) {
        if (Thread.currentThread().isInterrupted()) {
          throw new IOException("External FBIN scan interrupted");
        }
        buffer.clear();
        buffer.limit((int) Math.min(buffer.capacity(), remaining));
        int read = channel.read(buffer, position);
        if (read < 0) {
          throw new IOException("Unexpected EOF while scanning external FBIN " + path);
        }
        if (read == 0) {
          throw new IOException("Unable to make progress while scanning external FBIN " + path);
        }
        if (optionalDigest != null) {
          buffer.flip();
          optionalDigest.update(buffer);
        }
        position += read;
        remaining -= read;
        scanned += read;
        progress.accept(scanned);
      }
    }
    return length;
  }

  private static MessageDigest newSha256() {
    try {
      return MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException e) {
      throw new AssertionError("Every Java runtime must provide SHA-256", e);
    }
  }
}

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
import java.util.Arrays;
import java.util.HexFormat;
import java.util.Objects;

/**
 * A path-independent reference to a dense row-major float32 slice in an immutable FBIN file.
 *
 * <p>The complete-file SHA-256 digest is the content identity. The path is deliberately excluded so
 * an index cannot request arbitrary host files and can be relocated independently of the source. A
 * process must register an allowlisted local path for this content identity through {@link
 * ExternalFbinFileRegistry} before opening an index that references it.
 */
final class ExternalFbinReference {

  static final long HEADER_BYTES = 2L * Integer.BYTES;
  static final int SHA256_BYTES = 32;
  private static final HexFormat HEX = HexFormat.of();

  private final byte[] sha256;
  private final long fileLength;
  private final long payloadOffset;
  private final long payloadLength;
  private final int rows;
  private final int dimensions;

  private ExternalFbinReference(
      byte[] sha256,
      long fileLength,
      long payloadOffset,
      long payloadLength,
      int rows,
      int dimensions) {
    this.sha256 = validateSha256(sha256);
    if (fileLength <= HEADER_BYTES) {
      throw new IllegalArgumentException("FBIN file length must include a non-empty payload");
    }
    if (rows <= 0 || dimensions <= 0) {
      throw new IllegalArgumentException("rows and dimensions must be positive");
    }
    long rowBytes = Math.multiplyExact((long) dimensions, Float.BYTES);
    long expectedPayloadLength = Math.multiplyExact((long) rows, rowBytes);
    if (payloadLength != expectedPayloadLength) {
      throw new IllegalArgumentException(
          "payloadLength must equal rows * dimensions * 4; expected "
              + expectedPayloadLength
              + " but got "
              + payloadLength);
    }
    if (payloadOffset < HEADER_BYTES
        || Math.floorMod(payloadOffset - HEADER_BYTES, rowBytes) != 0) {
      throw new IllegalArgumentException("payloadOffset must be aligned to an FBIN row boundary");
    }
    if (Math.addExact(payloadOffset, payloadLength) > fileLength) {
      throw new IllegalArgumentException("Referenced payload extends beyond the FBIN file");
    }
    this.fileLength = fileLength;
    this.payloadOffset = payloadOffset;
    this.payloadLength = payloadLength;
    this.rows = rows;
    this.dimensions = dimensions;
  }

  /**
   * Reads and validates an FBIN header and creates a reference to a contiguous row range.
   *
   * <p>This method does not hash the file. {@code sha256Hex} must be a previously established digest
   * of the complete FBIN file.
   */
  static ExternalFbinReference fromFile(Path path, String sha256Hex, int firstRow, int rowCount)
      throws IOException {
    Objects.requireNonNull(path, "path");
    if (firstRow < 0 || rowCount <= 0) {
      throw new IllegalArgumentException("firstRow must be non-negative and rowCount positive");
    }

    try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
      ByteBuffer header = ByteBuffer.allocate((int) HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
      readFully(channel, header, 0L);
      header.flip();
      int fileRows = header.getInt();
      int dimensions = header.getInt();
      if (fileRows <= 0 || dimensions <= 0) {
        throw new IOException(
            "Invalid FBIN header in " + path + ": " + fileRows + " x " + dimensions);
      }
      if ((long) firstRow + rowCount > fileRows) {
        throw new IllegalArgumentException(
            "Requested row range ["
                + firstRow
                + ", "
                + ((long) firstRow + rowCount)
                + ") exceeds FBIN row count "
                + fileRows);
      }

      long rowBytes = Math.multiplyExact((long) dimensions, Float.BYTES);
      long expectedFileLength =
          Math.addExact(HEADER_BYTES, Math.multiplyExact((long) fileRows, rowBytes));
      long actualFileLength = channel.size();
      if (actualFileLength != expectedFileLength) {
        throw new IOException(
            "FBIN file length "
                + actualFileLength
                + " does not match header-derived length "
                + expectedFileLength
                + " for "
                + path);
      }

      long payloadOffset =
          Math.addExact(HEADER_BYTES, Math.multiplyExact((long) firstRow, rowBytes));
      long payloadLength = Math.multiplyExact((long) rowCount, rowBytes);
      return new ExternalFbinReference(
          parseSha256(sha256Hex),
          actualFileLength,
          payloadOffset,
          payloadLength,
          rowCount,
          dimensions);
    }
  }

  static ExternalFbinReference fromDescriptor(
      byte[] sha256,
      long fileLength,
      long payloadOffset,
      long payloadLength,
      int rows,
      int dimensions) {
    return new ExternalFbinReference(
        sha256, fileLength, payloadOffset, payloadLength, rows, dimensions);
  }

  static byte[] parseSha256(String sha256Hex) {
    Objects.requireNonNull(sha256Hex, "sha256Hex");
    String normalized =
        sha256Hex.regionMatches(true, 0, "sha256:", 0, "sha256:".length())
            ? sha256Hex.substring("sha256:".length())
            : sha256Hex;
    if (normalized.length() != SHA256_BYTES * 2) {
      throw new IllegalArgumentException("SHA-256 must contain exactly 64 hexadecimal characters");
    }
    try {
      return validateSha256(HEX.parseHex(normalized));
    } catch (IllegalArgumentException e) {
      throw new IllegalArgumentException("Invalid SHA-256 hexadecimal value", e);
    }
  }

  static String contentId(byte[] sha256) {
    return "sha256:" + HEX.formatHex(validateSha256(sha256));
  }

  private static byte[] validateSha256(byte[] sha256) {
    Objects.requireNonNull(sha256, "sha256");
    if (sha256.length != SHA256_BYTES) {
      throw new IllegalArgumentException("SHA-256 must contain exactly 32 bytes");
    }
    return sha256.clone();
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

  String contentId() {
    return contentId(sha256);
  }

  String sha256Hex() {
    return HEX.formatHex(sha256);
  }

  byte[] sha256() {
    return sha256.clone();
  }

  long fileLength() {
    return fileLength;
  }

  long payloadOffset() {
    return payloadOffset;
  }

  long payloadLength() {
    return payloadLength;
  }

  int rows() {
    return rows;
  }

  int dimensions() {
    return dimensions;
  }

  long firstRow() {
    return (payloadOffset - HEADER_BYTES) / ((long) dimensions * Float.BYTES);
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof ExternalFbinReference that)) {
      return false;
    }
    return fileLength == that.fileLength
        && payloadOffset == that.payloadOffset
        && payloadLength == that.payloadLength
        && rows == that.rows
        && dimensions == that.dimensions
        && Arrays.equals(sha256, that.sha256);
  }

  @Override
  public int hashCode() {
    int result = Arrays.hashCode(sha256);
    result = 31 * result + Long.hashCode(fileLength);
    result = 31 * result + Long.hashCode(payloadOffset);
    result = 31 * result + Long.hashCode(payloadLength);
    result = 31 * result + rows;
    result = 31 * result + dimensions;
    return result;
  }

  @Override
  public String toString() {
    return "ExternalFbinReference[contentId="
        + contentId()
        + ", firstRow="
        + firstRow()
        + ", rows="
        + rows
        + ", dimensions="
        + dimensions
        + "]";
  }
}

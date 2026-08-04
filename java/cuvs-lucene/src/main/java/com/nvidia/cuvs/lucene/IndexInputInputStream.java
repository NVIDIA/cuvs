/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.io.InputStream;
import org.apache.lucene.store.IndexInput;

/**
 * InputStream for reading from an IndexInput.
 *
 * @since 25.10
 */
final class IndexInputInputStream extends InputStream {

  final IndexInput in;
  long pos = 0;
  final long limit;

  /**
   * Initializes the {@link IndexInputInputStream}
   *
   * @param in instance of IndexInput
   */
  IndexInputInputStream(IndexInput in) {
    this.in = in;
    this.limit = in.length();
  }

  /**
   * Reads the next byte of data from the input stream.
   */
  @Override
  public int read() throws IOException {
    if (pos >= limit) {
      return -1;
    }
    pos++;
    return in.readByte();
  }

  /**
   * Reads up to len bytes of data from the input stream into an array of bytes.
   */
  @Override
  public int read(byte[] b, int off, int len) throws IOException {
    if (len <= 0) {
      return 0;
    }
    if (pos >= limit) {
      return -1;
    }
    long avail = limit - pos;
    if (len > avail) {
      len = (int) avail;
    }
    in.readBytes(b, off, len);
    pos += len;
    return len;
  }
}

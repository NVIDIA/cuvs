/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.io.OutputStream;
import org.apache.lucene.store.IndexOutput;

/**
 * OutputStream for writing into an IndexOutput
 *
 * @since 25.10
 */
final class IndexOutputOutputStream extends OutputStream {

  static final int DEFAULT_BUFFER_SIZE = 8192;

  final IndexOutput out;
  final int bufferSize;
  final byte[] buffer;
  int pos;

  /**
   * Initializes the {@link IndexOutputOutputStream}.
   *
   * @param out instance of IndexOutput
   */
  IndexOutputOutputStream(IndexOutput out) {
    this(out, DEFAULT_BUFFER_SIZE);
  }

  /**
   * Initializes the {@link IndexOutputOutputStream}.
   *
   * @param out instance of IndexOutput
   * @param bufferSize the size of buffer to use
   */
  IndexOutputOutputStream(IndexOutput out, int bufferSize) {
    this.out = out;
    this.bufferSize = bufferSize;
    this.buffer = new byte[bufferSize];
  }

  /**
   * Writes the specified byte to this output stream.
   */
  @Override
  public void write(int b) throws IOException {
    buffer[pos] = (byte) b;
    pos++;
    if (pos == bufferSize) {
      flush();
    }
  }

  /**
   * Writes len bytes from the specified byte array starting at offset off to this output stream.
   */
  @Override
  public void write(byte[] b, int offset, int length) throws IOException {
    if (pos != 0) {
      flush();
    }
    out.writeBytes(b, offset, length);
  }

  /**
   * Flushes this output stream and forces any buffered output bytes to be written out.
   */
  @Override
  public void flush() throws IOException {
    out.writeBytes(buffer, 0, pos);
    pos = 0;
  }

  /**
   * Closes this output stream and releases any system resources associated with this stream.
   */
  @Override
  public void close() throws IOException {
    this.flush();
  }
}

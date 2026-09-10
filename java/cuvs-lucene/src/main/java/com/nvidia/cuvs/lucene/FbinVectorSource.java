/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

/**
 * Prefetching, double-buffered {@link VectorSource} over an uncompressed {@code .fbin} file
 * ({@code [num_vectors int32][dim int32]} header, then contiguous little-endian float32 rows).
 *
 * <p>Opens the file ONCE. A background thread reads the (optionally sliced) range front-to-back
 * into two reusable direct buffers: while the caller consumes the current chunk, the reader fills
 * the next one, so the disk read overlaps with the caller's per-vector work. {@link #get(int,
 * float[])} unpacks directly into a caller-supplied array (no per-vector allocation).
 *
 * <p><b>Forward-only, single-consumer.</b> {@code get} must be called with strictly increasing
 * indices from a single thread, relative to the window this instance was constructed over — index 0 is
 * the first vector in that window, not necessarily the first vector in the file. To read only a
 * slice of a larger file (e.g. one segment of a partitioned build), construct with a {@code
 * [firstVector, count)} range so each instance streams just its own portion of the file.
 */
public final class FbinVectorSource implements VectorSource {

  private static final long HEADER_BYTES = 8;

  private static final class Chunk {
    final ByteBuffer buf;
    final long start;
    final int len;

    Chunk(ByteBuffer buf, long start, int len) {
      this.buf = buf;
      this.start = start;
      this.len = len;
    }
  }

  /** Sentinel placed on the ready queue once the reader has produced the final chunk. */
  private static final Chunk POISON = new Chunk(null, -1, 0);

  private final FileChannel channel;
  private final int dimension;
  private final int firstVector; // absolute index (in the file) of the first vector served
  private final int windowSize; // number of vectors this instance serves
  private final int vectorBytes;
  private final int chunkVectors;

  private final BlockingQueue<ByteBuffer> free = new ArrayBlockingQueue<>(2);
  private final BlockingQueue<Chunk> ready = new ArrayBlockingQueue<>(2);
  private final Thread reader;
  private volatile IOException readerError;

  private Chunk current; // consumer-owned; the chunk currently being served
  private long nextExpectedRelative; // forward-only guard, relative index

  /** Reads the whole file from index 0. */
  public FbinVectorSource(Path path, int chunkSizeMB) throws IOException {
    this(path, 0, -1, chunkSizeMB);
  }

  /**
   * Reads the contiguous range {@code [firstVector, firstVector + count)} of {@code path}, or to
   * end of file if {@code count <= 0}. {@link #get} then serves indices relative to {@code
   * firstVector} (0 = {@code firstVector}).
   */
  public FbinVectorSource(Path path, int firstVector, int count, int chunkSizeMB)
      throws IOException {
    this.channel = FileChannel.open(path, StandardOpenOption.READ);
    ByteBuffer header = ByteBuffer.allocate((int) HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
    readFully(header, 0);
    header.flip();
    int numVectors = header.getInt();
    this.dimension = header.getInt();
    this.vectorBytes = dimension * Float.BYTES;
    this.firstVector = firstVector;
    int endVector = count > 0 ? (int) Math.min((long) firstVector + count, numVectors) : numVectors;
    this.windowSize = Math.max(0, endVector - firstVector);

    long chunkBytes = (long) Math.max(1, chunkSizeMB) * 1024 * 1024;
    int cap = (Integer.MAX_VALUE - 16) / vectorBytes; // keep chunkVectors * vectorBytes in an int
    this.chunkVectors = (int) Math.max(1, Math.min(chunkBytes / vectorBytes, cap));

    // Two reusable direct buffers: the reader fills one while the consumer drains the other.
    for (int i = 0; i < 2; i++) {
      free.add(
          ByteBuffer.allocateDirect(chunkVectors * vectorBytes).order(ByteOrder.LITTLE_ENDIAN));
    }

    this.reader = new Thread(this::readLoop, "fbin-prefetch-reader");
    this.reader.setDaemon(true);
    this.reader.start();
  }

  @Override
  public int dimensions() {
    return dimension;
  }

  @Override
  public int size() {
    return windowSize;
  }

  /** Reader thread: fill chunks front-to-back, blocking on a free buffer between chunks. */
  private void readLoop() {
    long next = firstVector;
    long endVector = firstVector + windowSize;
    try {
      while (next < endVector) {
        ByteBuffer buf = free.take();
        int toRead = (int) Math.min(chunkVectors, endVector - next);
        buf.clear();
        buf.limit(toRead * vectorBytes);
        readFully(buf, HEADER_BYTES + next * (long) vectorBytes);
        ready.put(new Chunk(buf, next, toRead));
        next += toRead;
      }
      ready.put(POISON);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt(); // close() requested; stop quietly
    } catch (IOException e) {
      readerError = e;
      try {
        ready.put(POISON); // unblock the consumer so it can observe the error
      } catch (InterruptedException ignored) {
        Thread.currentThread().interrupt();
      }
    }
  }

  private void advance() throws IOException {
    if (current != null) {
      try {
        free.put(current.buf); // hand the drained buffer back to the reader
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new IOException("Interrupted returning chunk buffer", e);
      }
      current = null;
    }
    Chunk next;
    try {
      next = ready.take();
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IOException("Interrupted awaiting next chunk", e);
    }
    if (next == POISON) {
      if (readerError != null) {
        throw new IOException("Prefetch reader failed", readerError);
      }
      throw new IOException("No more chunks available (unexpected EOF in prefetch)");
    }
    current = next;
  }

  @Override
  public void get(int index, float[] dst) throws IOException {
    if (index < 0 || index >= windowSize) {
      throw new IndexOutOfBoundsException(
          "Index " + index + " out of bounds [0, " + windowSize + ")");
    }
    if (index < nextExpectedRelative) {
      throw new UnsupportedOperationException(
          "FbinVectorSource requires forward-only sequential access; got index "
              + index
              + " before previously served index "
              + (nextExpectedRelative - 1));
    }
    nextExpectedRelative = index + 1;
    long absolute = firstVector + index;
    while (current == null || absolute >= current.start + current.len) {
      advance();
    }
    int base = (int) (absolute - current.start) * vectorBytes;
    for (int i = 0; i < dimension; i++) {
      dst[i] = current.buf.getFloat(base + i * Float.BYTES);
    }
  }

  /** Convenience allocating variant (e.g. for a one-off query vector). */
  public float[] get(int index) throws IOException {
    float[] dst = new float[dimension];
    get(index, dst);
    return dst;
  }

  private void readFully(ByteBuffer buf, long position) throws IOException {
    long pos = position;
    while (buf.hasRemaining()) {
      int n = channel.read(buf, pos);
      if (n < 0) {
        throw new IOException("Unexpected EOF reading at position " + pos);
      }
      pos += n;
    }
  }

  @Override
  public void close() throws IOException {
    reader.interrupt();
    channel.close();
  }
}

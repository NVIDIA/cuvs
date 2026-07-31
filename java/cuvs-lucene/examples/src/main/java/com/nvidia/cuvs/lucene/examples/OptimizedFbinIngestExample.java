/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene.examples;

import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import com.nvidia.cuvs.lucene.AcceleratedHNSWParams;
import com.nvidia.cuvs.lucene.Lucene101AcceleratedHNSWCodec;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.logging.Logger;
import org.apache.commons.io.FileUtils;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * Reference pattern for efficiently ingesting a LARGE {@code .fbin} vector file into an accelerated
 * HNSW index. cuvs-lucene is a Lucene codec and sits below {@code addDocument}, so how you read your
 * source data is application code — this example shows the four properties that keep ingestion from
 * bottlenecking the GPU build:
 *
 * <ol>
 *   <li><b>Open once, read sequentially.</b> Reopening/seeking/closing the file per vector ("fd
 *       churn") is orders of magnitude slower than sequential reads. {@link PrefetchingFbinReader}
 *       opens the file ONCE and reads it front-to-back in large sequential chunks.
 *   <li><b>Bounded memory.</b> Pre-loading the whole file onto the JVM heap holds a redundant copy
 *       on top of Lucene's own per-segment buffer (~2x peak). This reader holds at most two chunks.
 *   <li><b>Overlap read with consumption.</b> A background reader thread fills the NEXT chunk (into a
 *       second buffer) while the ingest thread is still feeding the current chunk into {@code
 *       addDocument}, so the sequential disk read is hidden behind the per-document indexing work
 *       instead of serializing in front of it.
 *   <li><b>Reuse the vector array.</b> {@link PrefetchingFbinReader#get(int, float[])} unpacks
 *       directly into a caller-supplied array, avoiding a fresh {@code float[]} allocation per
 *       vector. This is safe because Lucene copies the vector value eagerly inside {@code
 *       addDocument} — the array may be reused as soon as {@code addDocument} returns.
 * </ol>
 *
 * <p>Adapt {@link PrefetchingFbinReader} to your own source (a DB, object store, or stream). The
 * properties above are what matter, not the {@code .fbin} specifics.
 *
 * <p>Usage: {@code OptimizedFbinIngestExample [<path-to.fbin>] [<chunkSizeMB>]}. With no arguments a
 * small demo {@code .fbin} is generated and indexed.
 */
public class OptimizedFbinIngestExample {

  private static final Logger log = Logger.getLogger(OptimizedFbinIngestExample.class.getName());
  private static final String ID_FIELD = "id";
  private static final String VECTOR_FIELD = "vector_field";

  public static void main(String[] args) throws Exception {
    int chunkSizeMB = args.length >= 2 ? Integer.parseInt(args[1]) : 32;
    Path indexDirPath = Paths.get(UUID.randomUUID().toString());

    Path fbinPath;
    boolean generated = false;
    if (args.length >= 1) {
      fbinPath = Paths.get(args[0]);
    } else {
      fbinPath = Paths.get("demo-" + UUID.randomUUID() + ".fbin");
      writeDemoFbin(fbinPath, 5000, 32, new Random(222));
      generated = true;
      log.info("No .fbin provided; generated a demo file at " + fbinPath);
    }

    try {
      buildIndex(fbinPath, indexDirPath, chunkSizeMB);
      runSampleSearch(indexDirPath, fbinPath, 5);
    } finally {
      FileUtils.deleteDirectory(indexDirPath.toFile());
      if (generated) {
        Files.deleteIfExists(fbinPath);
      }
    }
  }

  /**
   * Builds a single-segment accelerated HNSW index, streaming vectors from the {@code .fbin} in
   * sequential prefetched chunks (never holding more than two chunks in memory) and reusing a single
   * vector array across all documents.
   */
  private static void buildIndex(Path fbinPath, Path indexDirPath, int chunkSizeMB)
      throws Exception {
    AcceleratedHNSWParams params = new AcceleratedHNSWParams.Builder().build();
    Codec codec = new Lucene101AcceleratedHNSWCodec(params);

    try (PrefetchingFbinReader reader = new PrefetchingFbinReader(fbinPath, chunkSizeMB)) {
      int n = reader.size();

      // Keep the whole dataset in ONE segment: disable RAM-based flushing and raise the doc-count
      // flush threshold above the document count so nothing flushes before commit. (This only
      // controls Lucene's segment cadence; the vectors are not held here — they are streamed one
      // at a time from the chunked reader into addDocument.)
      // Order matters: enable the doc-count flush trigger BEFORE disabling the RAM trigger, since
      // Lucene rejects a config where both are disabled at once.
      IndexWriterConfig config =
          new IndexWriterConfig()
              .setCodec(codec)
              .setUseCompoundFile(false)
              .setMaxBufferedDocs(Math.max(2, n + 1))
              .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);

      log.info(
          "Indexing "
              + n
              + " vectors ("
              + reader.dimension()
              + "-dim) from "
              + fbinPath
              + " using "
              + chunkSizeMB
              + " MB prefetched sequential chunks");

      // One reusable array for the whole build — refilled per vector, copied eagerly by Lucene.
      float[] vector = new float[reader.dimension()];
      try (Directory dir = FSDirectory.open(indexDirPath);
          IndexWriter writer = new IndexWriter(dir, config)) {
        for (int i = 0; i < n; i++) {
          reader.get(i, vector); // sequential access -> served from the prefetched chunk, no alloc
          Document doc = new Document();
          doc.add(new StringField(ID_FIELD, Integer.toString(i), Field.Store.YES));
          doc.add(new KnnFloatVectorField(VECTOR_FIELD, vector, EUCLIDEAN));
          writer.addDocument(doc); // copies the vector -> 'vector' is safe to reuse next iteration
        }
        writer.commit(); // single flush -> single segment; the GPU CAGRA build happens here
      }
      log.info("Index build complete: " + indexDirPath);
    }
  }

  /** Runs one k-NN query using the first vector in the file to show the index is searchable. */
  private static void runSampleSearch(Path indexDirPath, Path fbinPath, int topK) throws Exception {
    float[] queryVector;
    try (PrefetchingFbinReader reader = new PrefetchingFbinReader(fbinPath, 1)) {
      queryVector = reader.get(0);
    }
    try (Directory dir = FSDirectory.open(indexDirPath);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      TopDocs results = searcher.search(new KnnFloatVectorQuery(VECTOR_FIELD, queryVector, topK), topK);
      log.info("Sample search returned " + results.scoreDocs.length + " hits:");
      for (int i = 0; i < results.scoreDocs.length; i++) {
        ScoreDoc sd = results.scoreDocs[i];
        String id = searcher.storedFields().document(sd.doc).get(ID_FIELD);
        log.info("  rank " + (i + 1) + ": id=" + id + " score=" + sd.score);
      }
    }
  }

  /** Writes a small random {@code .fbin} so the example is runnable without external data. */
  private static void writeDemoFbin(Path path, int numVectors, int dim, Random random)
      throws IOException {
    ByteBuffer buf =
        ByteBuffer.allocate(8 + numVectors * dim * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    buf.putInt(numVectors); // .fbin header: [num_vectors int32][dimension int32]
    buf.putInt(dim);
    for (int i = 0; i < numVectors; i++) {
      for (int j = 0; j < dim; j++) {
        buf.putFloat(random.nextFloat() * 100);
      }
    }
    buf.flip();
    try (FileChannel ch =
        FileChannel.open(
            path,
            StandardOpenOption.CREATE,
            StandardOpenOption.WRITE,
            StandardOpenOption.TRUNCATE_EXISTING)) {
      while (buf.hasRemaining()) {
        ch.write(buf);
      }
    }
  }

  /**
   * Prefetching, double-buffered sequential reader for uncompressed {@code .fbin} files ({@code
   * [num_vectors int32][dim int32]} header, then contiguous little-endian float32 rows).
   *
   * <p>Opens the file ONCE. A background thread reads the file front-to-back into two reusable direct
   * buffers: while the caller consumes the current chunk, the reader fills the next one, so the disk
   * read overlaps with the caller's per-vector work. {@link #get(int, float[])} unpacks directly into
   * a caller-supplied array (no per-vector allocation).
   *
   * <p><b>Forward-only, single-consumer.</b> {@code get} must be called with non-decreasing indices
   * from a single thread — the intended pattern for streaming ingestion.
   */
  static final class PrefetchingFbinReader implements AutoCloseable {

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
    private final int vectorCount;
    private final int vectorBytes;
    private final int chunkVectors;

    private final BlockingQueue<ByteBuffer> free = new ArrayBlockingQueue<>(2);
    private final BlockingQueue<Chunk> ready = new ArrayBlockingQueue<>(2);
    private final Thread reader;
    private volatile IOException readerError;

    private Chunk current; // consumer-owned; the chunk currently being served

    PrefetchingFbinReader(Path path, int chunkSizeMB) throws IOException {
      this.channel = FileChannel.open(path, StandardOpenOption.READ);
      ByteBuffer header = ByteBuffer.allocate((int) HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
      readFully(header, 0);
      header.flip();
      this.vectorCount = header.getInt();
      this.dimension = header.getInt();
      this.vectorBytes = dimension * Float.BYTES;

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

    int size() {
      return vectorCount;
    }

    int dimension() {
      return dimension;
    }

    /** Reader thread: fill chunks front-to-back, blocking on a free buffer between chunks. */
    private void readLoop() {
      long next = 0;
      try {
        while (next < vectorCount) {
          ByteBuffer buf = free.take();
          int toRead = (int) Math.min(chunkVectors, vectorCount - next);
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

    /** Fills {@code dst} with the vector at {@code index} (no allocation). */
    void get(int index, float[] dst) throws IOException {
      if (index < 0 || index >= vectorCount) {
        throw new IndexOutOfBoundsException(
            "Index " + index + " out of bounds [0, " + vectorCount + ")");
      }
      while (current == null || index >= current.start + current.len) {
        advance();
      }
      if (index < current.start) {
        throw new UnsupportedOperationException(
            "PrefetchingFbinReader requires forward-only sequential access; got index "
                + index
                + " before current chunk start "
                + current.start);
      }
      int base = (int) (index - current.start) * vectorBytes;
      for (int i = 0; i < dimension; i++) {
        dst[i] = current.buf.getFloat(base + i * Float.BYTES);
      }
    }

    /** Convenience allocating variant (e.g. for a one-off query vector). */
    float[] get(int index) throws IOException {
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
}

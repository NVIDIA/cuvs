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
 * Reference pattern for ingesting a LARGE {@code .fbin} vector file into an accelerated HNSW index
 * without the two pitfalls that dominate build time and memory when loading big vector files:
 *
 * <ol>
 *   <li><b>fd churn</b> — reopening/seeking/closing the file per vector is orders of magnitude
 *       slower than sequential reads. {@link ChunkedFbinReader} opens the file ONCE and reads it
 *       front-to-back in large sequential chunks.
 *   <li><b>unbounded / doubled memory</b> — pre-loading the whole file into the JVM heap holds a
 *       redundant copy on top of Lucene's own per-segment buffer (~2x peak). The chunked reader
 *       holds at most one chunk, streaming each vector straight into {@code addDocument}.
 * </ol>
 *
 * <p>cuvs-lucene is a Lucene codec and sits below {@code addDocument}, so how you read your source
 * data is application code — adapt {@link ChunkedFbinReader} to your own source (a DB, object
 * store, or stream). The properties that matter are: <em>open once, read sequentially, bound memory
 * to a chunk</em>.
 *
 * <p>Usage: {@code ChunkedFbinIngestExample [<path-to.fbin>] [<chunkSizeMB>]}. With no arguments a
 * small demo {@code .fbin} is generated and indexed.
 */
public class ChunkedFbinIngestExample {

  private static final Logger log = Logger.getLogger(ChunkedFbinIngestExample.class.getName());
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
   * sequential chunks (never holding more than one chunk in memory).
   */
  private static void buildIndex(Path fbinPath, Path indexDirPath, int chunkSizeMB)
      throws Exception {
    AcceleratedHNSWParams params = new AcceleratedHNSWParams.Builder().build();
    Codec codec = new Lucene101AcceleratedHNSWCodec(params);

    try (ChunkedFbinReader reader = new ChunkedFbinReader(fbinPath, chunkSizeMB)) {
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
              + " MB sequential chunks");

      try (Directory dir = FSDirectory.open(indexDirPath);
          IndexWriter writer = new IndexWriter(dir, config)) {
        for (int i = 0; i < n; i++) {
          float[] vector = reader.get(i); // sequential access -> served from the current chunk
          Document doc = new Document();
          doc.add(new StringField(ID_FIELD, Integer.toString(i), Field.Store.YES));
          doc.add(new KnnFloatVectorField(VECTOR_FIELD, vector, EUCLIDEAN));
          writer.addDocument(doc);
        }
        writer.commit(); // single flush -> single segment; the GPU CAGRA build happens here
      }
      log.info("Index build complete: " + indexDirPath);
    }
  }

  /** Runs one k-NN query using the first vector in the file to show the index is searchable. */
  private static void runSampleSearch(Path indexDirPath, Path fbinPath, int topK) throws Exception {
    float[] queryVector;
    try (ChunkedFbinReader reader = new ChunkedFbinReader(fbinPath, 1)) {
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
   * Chunked, sequential reader for uncompressed {@code .fbin} files
   * ({@code [num_vectors int32][dim int32]} header, then contiguous little-endian float32 rows).
   *
   * <p>Opens the file ONCE and serves {@link #get(int)} from a reusable buffer that is refilled
   * with a single large sequential read whenever the requested index leaves the current chunk. For
   * sequential access (index = 0, 1, 2, ...) it reads the file front-to-back in {@code size/chunk}
   * bulk reads while holding only one chunk in memory — the opposite of reopening the file per
   * vector.
   */
  static final class ChunkedFbinReader implements AutoCloseable {

    private static final long HEADER_BYTES = 8;

    private final FileChannel channel;
    private final int dimension;
    private final int vectorCount;
    private final int vectorBytes;
    private final int chunkVectors;
    private final ByteBuffer chunkBuffer;

    private long chunkStart = -1; // first vector index currently buffered
    private int chunkLen = 0; // number of vectors currently buffered

    ChunkedFbinReader(Path path, int chunkSizeMB) throws IOException {
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
      this.chunkBuffer =
          ByteBuffer.allocateDirect(chunkVectors * vectorBytes).order(ByteOrder.LITTLE_ENDIAN);
    }

    int size() {
      return vectorCount;
    }

    int dimension() {
      return dimension;
    }

    float[] get(int index) throws IOException {
      if (index < 0 || index >= vectorCount) {
        throw new IndexOutOfBoundsException(
            "Index " + index + " out of bounds [0, " + vectorCount + ")");
      }
      if (chunkStart < 0 || index < chunkStart || index >= chunkStart + chunkLen) {
        long start = (index / (long) chunkVectors) * chunkVectors;
        int toRead = (int) Math.min(chunkVectors, vectorCount - start);
        chunkBuffer.clear();
        chunkBuffer.limit(toRead * vectorBytes);
        readFully(chunkBuffer, HEADER_BYTES + start * (long) vectorBytes);
        chunkStart = start;
        chunkLen = toRead;
      }
      int base = (int) (index - chunkStart) * vectorBytes;
      float[] vector = new float[dimension];
      for (int i = 0; i < dimension; i++) {
        vector[i] = chunkBuffer.getFloat(base + i * Float.BYTES);
      }
      return vector;
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
      channel.close();
    }
  }
}

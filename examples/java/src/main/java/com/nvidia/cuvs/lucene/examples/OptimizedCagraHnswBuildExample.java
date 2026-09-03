/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene.examples;

import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import com.nvidia.cuvs.lucene.AcceleratedHNSWParams;
import com.nvidia.cuvs.lucene.CagraHnswBulkIndexWriter;
import com.nvidia.cuvs.lucene.FbinVectorSource;
import com.nvidia.cuvs.spi.CuVSProvider;
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
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * Reference usage of {@link CagraHnswBulkIndexWriter}: builds an accelerated HNSW index (whose
 * graph is built on the GPU with CAGRA) from a {@code .fbin} vector file on local disk.
 *
 * <p>Demonstrates both ways to use {@link CagraHnswBulkIndexWriter}:
 *
 * <ul>
 *   <li>{@link #main} — the one-shot convenience path. All of the bulk-build mechanics —
 *       prefetched streaming reads, native flat buffering, the {@code IndexWriterConfig} tuning
 *       that guarantees a single unmerged segment per slice, K-segment partitioning, and combining
 *       the result by hardlink — are owned by {@link CagraHnswBulkIndexWriter} itself; see its
 *       Javadoc for how each of those works and the tradeoffs of {@code numSegments} and {@code
 *       overlap}. This example only wires up what's genuinely application-specific: where the
 *       vectors come from ({@link FbinVectorSource}, or your own {@link
 *       com.nvidia.cuvs.lucene.VectorSource} for a different data source), the graph-build quality
 *       knobs ({@link AcceleratedHNSWParams}), and — via a {@link
 *       CagraHnswBulkIndexWriter.FieldCallback} — any per-vector metadata to attach.
 *   <li>{@link #runManualExample} — the manual, direct-instance path: construct a {@link
 *       CagraHnswBulkIndexWriter} yourself and drive {@code addDocument}/{@code close} exactly
 *       like a plain Lucene {@code IndexWriter}, building each {@link Document} (metadata included)
 *       yourself instead of going through a callback.
 * </ul>
 *
 * <p>Usage: {@code OptimizedCagraHnswBuildExample [<path-to.fbin>] [<chunkSizeMB>] [<numSegments>]
 * [<overlap:true|false>]}. With no arguments a small demo {@code .fbin} is generated and indexed as a
 * single segment.
 */
public class OptimizedCagraHnswBuildExample {

  private static final Logger log =
      Logger.getLogger(OptimizedCagraHnswBuildExample.class.getName());
  private static final String ID_FIELD = "id";
  private static final String CATEGORY_FIELD = "category";
  private static final String VECTOR_FIELD = "vector_field";

  public static void main(String[] args) throws Exception {
    // It is recommended to enable RMM allocation mode at application start, before constructing
    // any CagraHnswBulkIndexWriter, to avoid device-wide sync from the default allocator.
    CuVSProvider.provider().enableRMMAsyncMemory();

    int chunkSizeMB = args.length >= 2 ? Integer.parseInt(args[1]) : 32;
    int numSegments = args.length >= 3 ? Math.max(1, Integer.parseInt(args[2])) : 1;
    boolean overlap = args.length >= 4 && Boolean.parseBoolean(args[3]);
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
      int dim;
      try (FbinVectorSource probe = new FbinVectorSource(fbinPath, 1)) {
        dim = probe.dimensions();
      }

      CagraHnswBulkIndexWriter.Config config =
          CagraHnswBulkIndexWriter.Config.builder()
              .field(VECTOR_FIELD, dim, VectorSimilarityFunction.EUCLIDEAN)
              .idField(ID_FIELD)
              .graphBuild(
                  new AcceleratedHNSWParams.Builder()
                      // HEURISTIC lets cuVS pick the build algorithm and auto-tune its parameters
                      // based on maxConn and beamWidth below.
                      .withStrategy(AcceleratedHNSWParams.Strategy.HEURISTIC)
                      // Primary recall/graph-size knobs. Higher values improve recall at the cost
                      // of a larger graph and longer build. Match to your dataset and recall
                      // target.
                      .withMaxConn(32)
                      .withBeamWidth(32)
                      // Must match the distance metric used when querying the index.
                      .withCuvsDistanceType(CuvsDistanceType.L2Expanded)
                      // Starting point: one thread per logical CPU. Profile and tune for your
                      // hardware.
                      .withWriterThreads(Runtime.getRuntime().availableProcessors())
                      .build())
              .segments(numSegments, overlap)
              .targetDirectory(indexDirPath)
              .build();

      log.info(
          "Indexing "
              + fbinPath
              + " ("
              + dim
              + "-dim) into "
              + numSegments
              + " segment(s), "
              + (overlap && numSegments > 1 ? "overlapped" : "sequential")
              + " build, "
              + chunkSizeMB
              + " MB prefetched chunks");

      // FieldCallback lets the one-shot path attach metadata per vector: indexFbin/build build the
      // id+vector fields internally (they own the loop), so this is how a caller reaches the
      // Document to add anything else -- here, an illustrative "even"/"odd" category by id.
      CagraHnswBulkIndexWriter.indexFbin(
          fbinPath,
          config,
          (doc, id) ->
              doc.add(
                  new StringField(CATEGORY_FIELD, id % 2 == 0 ? "even" : "odd", Field.Store.YES)),
          chunkSizeMB);
      log.info("Index build complete: " + indexDirPath);

      runSampleSearch(indexDirPath, fbinPath, 5);
    } finally {
      FileUtils.deleteDirectory(indexDirPath.toFile());
      if (generated) {
        Files.deleteIfExists(fbinPath);
      }
    }

    runManualExample();
  }

  /** Runs one k-NN query using the first vector in the file to show the index is searchable. */
  private static void runSampleSearch(Path indexDirPath, Path fbinPath, int topK) throws Exception {
    float[] queryVector;
    try (FbinVectorSource reader = new FbinVectorSource(fbinPath, 1)) {
      queryVector = reader.get(0);
    }
    try (Directory dir = FSDirectory.open(indexDirPath);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      TopDocs results =
          searcher.search(new KnnFloatVectorQuery(VECTOR_FIELD, queryVector, topK), topK);
      log.info("Sample search returned " + results.scoreDocs.length + " hits:");
      for (int i = 0; i < results.scoreDocs.length; i++) {
        ScoreDoc sd = results.scoreDocs[i];
        Document hit = searcher.storedFields().document(sd.doc);
        log.info(
            "  rank "
                + (i + 1)
                + ": id="
                + hit.get(ID_FIELD)
                + " category="
                + hit.get(CATEGORY_FIELD)
                + " score="
                + sd.score);
      }
    }
  }

  /**
   * Short demonstration of the manual, direct-instance API: {@link CagraHnswBulkIndexWriter} is
   * constructed directly and driven with {@code addDocument}/{@code close}, the same shape as a
   * plain Lucene {@code IndexWriter} — the caller builds each {@link Document} itself, including
   * whatever metadata it wants, with no callback needed since it already owns the loop. Unlike the
   * one-shot path above, this only ever builds a single segment; K-segment partitioning and
   * overlap are only available via {@link CagraHnswBulkIndexWriter#indexFbin}/{@link
   * CagraHnswBulkIndexWriter#build}.
   */
  private static void runManualExample() throws Exception {
    int numDocs = 200;
    int dim = 16;
    Random random = new Random(7);
    Path manualIndexDirPath = Paths.get("manual-" + UUID.randomUUID());

    try {
      CagraHnswBulkIndexWriter.Config config =
          CagraHnswBulkIndexWriter.Config.builder()
              .field(VECTOR_FIELD, dim, VectorSimilarityFunction.EUCLIDEAN)
              .graphBuild(new AcceleratedHNSWParams.Builder().build())
              .build();

      float[][] vectors = new float[numDocs][dim];
      try (Directory dir = FSDirectory.open(manualIndexDirPath);
          CagraHnswBulkIndexWriter writer =
              new CagraHnswBulkIndexWriter(dir, new IndexWriterConfig(), config, numDocs)) {
        for (int i = 0; i < numDocs; i++) {
          for (int j = 0; j < dim; j++) {
            vectors[i][j] = random.nextFloat() * 100;
          }
          Document doc = new Document();
          doc.add(new StringField(ID_FIELD, Integer.toString(i), Field.Store.YES));
          doc.add(new StringField(CATEGORY_FIELD, i % 2 == 0 ? "even" : "odd", Field.Store.YES));
          doc.add(
              new KnnFloatVectorField(
                  VECTOR_FIELD, vectors[i], VectorSimilarityFunction.EUCLIDEAN));
          writer.addDocument(doc); // same call shape as a plain IndexWriter
        }
      } // close() runs the single native-buffered flush (the GPU CAGRA build happens here)

      try (Directory dir = FSDirectory.open(manualIndexDirPath);
          DirectoryReader reader = DirectoryReader.open(dir)) {
        IndexSearcher searcher = new IndexSearcher(reader);
        TopDocs results = searcher.search(new KnnFloatVectorQuery(VECTOR_FIELD, vectors[0], 1), 1);
        Document hit = searcher.storedFields().document(results.scoreDocs[0].doc);
        log.info(
            "Manual example: nearest neighbor of vector 0 is id="
                + hit.get(ID_FIELD)
                + " category="
                + hit.get(CATEGORY_FIELD));
      }
    } finally {
      FileUtils.deleteDirectory(manualIndexDirPath.toFile());
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
}

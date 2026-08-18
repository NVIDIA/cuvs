/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene.examples;

import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import com.nvidia.cuvs.lucene.AcceleratedHNSWParams;
import com.nvidia.cuvs.lucene.Lucene101AcceleratedHNSWCodec;
import com.nvidia.cuvs.spi.CuVSProvider;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Semaphore;
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
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.misc.store.HardlinkCopyDirectoryWrapper;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * Reference pattern for building an accelerated HNSW index (whose graph is built on the GPU with
 * CAGRA) from a LARGE {@code .fbin} vector file with <b>every optimization turned on</b> — spanning
 * both how vectors are ingested and how the GPU build is configured and partitioned. cuvs-lucene is a
 * Lucene codec that sits below {@code addDocument}, so how you read your source data and how you shape
 * segments is application code — this example shows the full recipe.
 *
 * <h2>The knobs, and why each one matters</h2>
 *
 * <ol>
 *   <li><b>Streaming, prefetched, bounded-memory reads.</b> {@link PrefetchingFbinReader} opens the
 *       file ONCE, reads it front-to-back in large sequential chunks (no per-vector seek/close "fd
 *       churn"), holds at most two chunks (no whole-file heap copy), fills the NEXT chunk on a
 *       background thread while the ingest thread drains the current one (disk read hidden behind
 *       indexing), and unpacks into a caller-reused {@code float[]} (no per-vector allocation — safe
 *       because Lucene copies the value eagerly inside {@code addDocument}).
 *   <li><b>Native flat buffering.</b> {@link AcceleratedHNSWParams.Builder#withNumInputVectors} sizes
 *       a native host matrix to exactly the segment's vector count, so vectors stream straight into
 *       the buffer the GPU build consumes instead of piling up as a {@code List<float[]>} on the JVM
 *       heap that must then be assembled. This removes the assembly copy and roughly halves peak host
 *       memory. It requires a <b>single-segment</b> build (no flush before commit, no merge).
 *   <li><b>Automatic graph-build algorithm.</b> {@code HEURISTIC} strategy defaults to
 *       {@code AUTO_SELECT}, letting cuVS pick the CAGRA build algorithm by dataset size (NN_DESCENT
 *       below ~5M vectors, IVF_PQ at or above) and auto-tune its parameters. NN_DESCENT generally
 *       reaches higher recall but takes longer to build; IVF_PQ is faster but slightly lower recall.
 *       Force one explicitly via {@code withCagraGraphBuildAlgo} only under expert guidance.
 *   <li><b>Partitioned multi-segment build.</b> Because native flat buffering is single-segment, "K
 *       segments" means K independent single-segment builds over contiguous slices, combined at the
 *       end. This is a deliberate memory/throughput lever — see the assumptions below.
 * </ol>
 *
 * <h2>Assumptions for the user-specified number of segments</h2>
 *
 * <p>{@code numSegments} is <b>your</b> choice and it is a trade-off, not a free speedup:
 *
 * <ul>
 *   <li>Each segment is one native-flat build over a {@code 1/K} slice, so <b>peak host memory scales
 *       as {@code 1/K}</b> (sequential mode) — this is the point of partitioning for large or
 *       memory-bounded datasets.
 *   <li>The <b>GPU is serialized</b>: only one CAGRA build runs on the device at a time. Extra
 *       segments do NOT parallelize the graph build; they only reduce host memory and (in overlap
 *       mode) let one segment's host-side ingest run during a prior segment's GPU commit.
 *   <li>Search fans out across all K segment graphs, so more segments trade a little query throughput
 *       (and can shift recall) for lower build-time memory. Pick K to fit your memory budget, not
 *       higher.
 *   <li>Each slice must stay a single segment: {@code maxBufferedDocs > sliceSize} and {@link
 *       NoMergePolicy} (no flush, no merge) so native flat buffering stays valid.
 * </ul>
 *
 * <h2>Sequential vs overlapped</h2>
 *
 * <p>With {@code overlap=false} the K slices are built as K sequential passes appended to one
 * directory; peak host memory is a single slice's buffer ({@code N/K}). With {@code overlap=true} a
 * bounded pool builds up to {@code PIPELINE_DEPTH} segments at once into their own directories, so a
 * segment's ingest overlaps a prior segment's (serialized) GPU commit; the finished per-segment
 * indexes are then combined by <b>hardlinking</b> their files into the final directory ({@link
 * HardlinkCopyDirectoryWrapper} + {@code addIndexes}, no bulk copy of the vector data). Overlap costs
 * up to {@code PIPELINE_DEPTH * (N/K)} peak host memory and can hide most of the ingest time behind
 * the GPU build, bounded by disk contention between concurrent reads and the committing segment's
 * writes.
 *
 * <p>Adapt {@link PrefetchingFbinReader} to your own source (a DB, object store, or stream). The
 * properties above are what matter, not the {@code .fbin} specifics.
 *
 * <p>Usage: {@code OptimizedCagraHnswBuildExample [<path-to.fbin>] [<chunkSizeMB>] [<numSegments>]
 * [<overlap:true|false>]}. With no arguments a small demo {@code .fbin} is generated and indexed as a
 * single segment.
 */
public class OptimizedCagraHnswBuildExample {

  private static final Logger log =
      Logger.getLogger(OptimizedCagraHnswBuildExample.class.getName());
  private static final String ID_FIELD = "id";
  private static final String VECTOR_FIELD = "vector_field";

  /** Max segments built concurrently in overlap mode; peak host memory is this many slice buffers. */
  private static final int PIPELINE_DEPTH = 2;

  public static void main(String[] args) throws Exception {
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

    // Must not be called when using a CPU-only Lucene codec: those paths never load the cuVS
    // native library, so CuVSProvider resolves to UnsupportedProvider and throws.
    CuVSProvider.provider().enableRMMAsyncMemory();
    try {
      buildIndex(fbinPath, indexDirPath, chunkSizeMB, numSegments, overlap);
      runSampleSearch(indexDirPath, fbinPath, 5);
    } finally {
      FileUtils.deleteDirectory(indexDirPath.toFile());
      if (generated) {
        Files.deleteIfExists(fbinPath);
      }
    }
  }

  /**
   * Builds an accelerated HNSW index from {@code fbinPath} into {@code numSegments} contiguous
   * slices, dispatching to the sequential or overlapped partitioned build. {@code numSegments == 1}
   * is the plain single-segment native-flat build.
   */
  private static void buildIndex(
      Path fbinPath, Path indexDirPath, int chunkSizeMB, int numSegments, boolean overlap)
      throws Exception {
    int total;
    int dim;
    try (PrefetchingFbinReader reader = new PrefetchingFbinReader(fbinPath, 1)) {
      total = reader.size();
      dim = reader.dimension();
    }
    List<int[]> slices = sliceEvenly(total, numSegments); // [start, size] per segment

    log.info(
        "Indexing "
            + total
            + " vectors ("
            + dim
            + "-dim) from "
            + fbinPath
            + " into "
            + slices.size()
            + " segment(s), "
            + (overlap && slices.size() > 1 ? "overlapped" : "sequential")
            + " build, "
            + chunkSizeMB
            + " MB prefetched chunks");

    if (overlap && slices.size() > 1) {
      buildOverlapped(fbinPath, indexDirPath, chunkSizeMB, slices, dim);
    } else {
      buildSequential(fbinPath, indexDirPath, chunkSizeMB, slices, dim);
    }
    log.info("Index build complete: " + indexDirPath);
  }

  /**
   * Sequential partitioned build: one forward-only prefetch reader streamed front-to-back across all
   * K slices, each slice built as a single native-flat segment appended to the same directory (first
   * pass {@code CREATE}, later passes {@code APPEND}). Peak host memory is one slice's native buffer.
   */
  private static void buildSequential(
      Path fbinPath, Path indexDirPath, int chunkSizeMB, List<int[]> slices, int dim)
      throws Exception {
    float[] scratch = new float[dim]; // one reusable array for the whole build
    try (PrefetchingFbinReader reader = new PrefetchingFbinReader(fbinPath, chunkSizeMB);
        Directory dir = FSDirectory.open(indexDirPath)) {
      for (int p = 0; p < slices.size(); p++) {
        int[] slice = slices.get(p);
        log.info(
            "Building segment "
                + (p + 1)
                + "/"
                + slices.size()
                + ": docs ["
                + slice[0]
                + ", "
                + (slice[0] + slice[1])
                + ")");
        buildSegment(dir, reader, scratch, slice[0], slice[1], p == 0, null);
      }
    }
  }

  /**
   * Overlapped partitioned build: a bounded pool builds up to {@link #PIPELINE_DEPTH} segments at
   * once, each into its OWN directory with its OWN prefetch reader over just its slice, so a
   * segment's ingest overlaps a prior segment's GPU commit. The GPU build is serialized on a single
   * permit. The finished per-segment indexes are then hardlinked into the final directory (no bulk
   * copy of the vector data).
   */
  private static void buildOverlapped(
      Path fbinPath, Path indexDirPath, int chunkSizeMB, List<int[]> slices, int dim)
      throws Exception {
    int depth = Math.min(slices.size(), PIPELINE_DEPTH);
    int maxSlice = slices.stream().mapToInt(s -> s[1]).max().orElse(0);
    double peakHostGb = (double) depth * maxSlice * dim * Float.BYTES / 1e9;
    log.info(
        "Overlapped build: "
            + slices.size()
            + " segment(s), pipeline depth "
            + depth
            + " (up to "
            + depth
            + " co-resident native host buffers, ~"
            + String.format("%.2f", peakHostGb)
            + " GB peak host)");

    List<Path> segDirs = new ArrayList<>();
    for (int p = 0; p < slices.size(); p++) {
      segDirs.add(Paths.get(indexDirPath + "_p" + p));
    }
    // Start from fresh per-segment temp dirs, and always remove them afterwards (even on failure)
    // so
    // a crashed build does not leave orphaned per-segment indexes behind.
    for (Path segDir : segDirs) {
      FileUtils.deleteQuietly(segDir.toFile());
    }
    try {
      Semaphore gpuPermit = new Semaphore(1); // serialize the GPU CAGRA build across segments
      ExecutorService pool = Executors.newFixedThreadPool(depth);
      List<Future<?>> futures = new ArrayList<>();
      for (int p = 0; p < slices.size(); p++) {
        int[] slice = slices.get(p);
        Path segDir = segDirs.get(p);
        futures.add(
            pool.submit(
                () -> {
                  float[] scratch = new float[dim];
                  try (PrefetchingFbinReader reader =
                          new PrefetchingFbinReader(fbinPath, slice[0], slice[1], chunkSizeMB);
                      Directory d = FSDirectory.open(segDir)) {
                    // createNew=true: each segment is a fresh single-segment index in its own dir.
                    buildSegment(d, reader, scratch, slice[0], slice[1], true, gpuPermit);
                  }
                  return null;
                }));
      }
      pool.shutdown();
      try {
        for (Future<?> f : futures) {
          f.get(); // propagate any build failure
        }
      } finally {
        pool.shutdownNow();
      }
      combineByHardlink(indexDirPath, segDirs);
    } finally {
      for (Path segDir : segDirs) {
        FileUtils.deleteQuietly(segDir.toFile());
      }
    }
  }

  /**
   * Builds one segment from the contiguous slice {@code [start, start + size)}: a single-threaded
   * {@link IndexWriter} whose codec's {@code numInputVectors} is sized to the slice (native flat
   * buffering, single segment). When {@code gpuPermit} is non-null the commit — which runs the GPU
   * CAGRA build — is serialized on it while other segments' host-side ingest may proceed.
   */
  private static void buildSegment(
      Directory dir,
      PrefetchingFbinReader reader,
      float[] scratch,
      int start,
      int size,
      boolean createNew,
      Semaphore gpuPermit)
      throws Exception {
    Codec codec = codecFor(size);

    // Keep this slice in ONE segment: raise the doc-count flush threshold above the slice size and
    // disable RAM-based flushing so nothing flushes before commit, and forbid merges. This is what
    // makes native flat buffering valid. Order matters: enable the doc-count trigger BEFORE
    // disabling
    // the RAM trigger, since Lucene rejects a config where both are disabled at once.
    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(codec)
            .setUseCompoundFile(false)
            .setMaxBufferedDocs(Math.max(2, size + 1))
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH)
            .setMergePolicy(NoMergePolicy.INSTANCE)
            .setOpenMode(
                createNew ? IndexWriterConfig.OpenMode.CREATE : IndexWriterConfig.OpenMode.APPEND);

    try (IndexWriter writer = new IndexWriter(dir, config)) {
      for (int i = 0; i < size; i++) {
        int id = start + i;
        reader.get(id, scratch); // sequential access -> served from the prefetched chunk, no alloc
        Document doc = new Document();
        doc.add(new StringField(ID_FIELD, Integer.toString(id), Field.Store.YES));
        doc.add(new KnnFloatVectorField(VECTOR_FIELD, scratch, EUCLIDEAN));
        writer.addDocument(doc); // copies the vector -> 'scratch' is safe to reuse next iteration
      }
      // The single flush at commit is where the GPU CAGRA build runs; serialize it if asked.
      if (gpuPermit != null) {
        gpuPermit.acquire();
        try {
          writer.commit();
        } finally {
          gpuPermit.release();
        }
      } else {
        writer.commit();
      }
    }
  }

  /**
   * Combines the per-segment indexes into {@code indexDirPath} by hardlinking their files (same
   * filesystem) rather than copying the vector data. {@link HardlinkCopyDirectoryWrapper} falls back
   * to a byte copy automatically if the segment dirs and the final dir are on different filesystems.
   */
  private static void combineByHardlink(Path indexDirPath, List<Path> segDirs) throws IOException {
    Directory[] sources = new Directory[segDirs.size()];
    try {
      for (int i = 0; i < segDirs.size(); i++) {
        sources[i] = FSDirectory.open(segDirs.get(i));
      }
      IndexWriterConfig iwc =
          new IndexWriterConfig().setMergePolicy(NoMergePolicy.INSTANCE); // keep segments separate
      try (Directory target = new HardlinkCopyDirectoryWrapper(FSDirectory.open(indexDirPath));
          IndexWriter combiner = new IndexWriter(target, iwc)) {
        combiner.addIndexes(sources);
      }
    } finally {
      for (Directory s : sources) {
        if (s != null) {
          s.close();
        }
      }
    }
  }

  /** Builds a codec with all knobs on, sizing the native flat buffer to {@code numInputVectors}. */
  private static Codec codecFor(int numInputVectors) throws Exception {
    AcceleratedHNSWParams params =
        new AcceleratedHNSWParams.Builder()
            // HEURISTIC lets cuVS pick the build algorithm and auto-tune its parameters based on
            // maxConn and beamWidth below.
            .withStrategy(AcceleratedHNSWParams.Strategy.HEURISTIC)
            // Primary recall/graph-size knobs. Higher values improve recall at the cost of a
            // larger graph and longer build. Match to your dataset and recall target.
            .withMaxConn(32)
            .withBeamWidth(32)
            // Must match the distance metric used when querying the index.
            .withCuvsDistanceType(CuvsDistanceType.L2Expanded)
            // Starting point: one thread per logical CPU. Profile and tune for your hardware.
            .withWriterThreads(Runtime.getRuntime().availableProcessors())
            // Native flat buffering: the value MUST equal the number of vectors actually ingested
            // into this segment; the writer fails fast if they differ. If some input vectors are
            // excluded (e.g. filtered during ingest), either pre-count the survivors in a separate
            // pass or omit withNumInputVectors (pass 0) to fall back to the heap-buffered path,
            // which buffers all vectors in a List<float[]> on the JVM heap before building and
            // therefore uses more peak host memory. Index-sorted segments (IndexWriterConfig
            // .setIndexSort) are also unsupported, and binary/scalar quantized fields are not yet
            // supported.
            .withNumInputVectors(numInputVectors)
            .build();
    return new Lucene101AcceleratedHNSWCodec(params);
  }

  /** Splits {@code total} into {@code k} contiguous [start, size] slices, spreading the remainder. */
  private static List<int[]> sliceEvenly(int total, int k) {
    List<int[]> slices = new ArrayList<>();
    int base = total / k;
    int rem = total % k;
    int start = 0;
    for (int p = 0; p < k; p++) {
      int size = base + (p < rem ? 1 : 0); // spread the remainder over the first slices
      if (size <= 0) {
        continue;
      }
      slices.add(new int[] {start, size});
      start += size;
    }
    return slices;
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
      TopDocs results =
          searcher.search(new KnnFloatVectorQuery(VECTOR_FIELD, queryVector, topK), topK);
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
   * <p>Opens the file ONCE. A background thread reads the (optionally sliced) range front-to-back
   * into two reusable direct buffers: while the caller consumes the current chunk, the reader fills
   * the next one, so the disk read overlaps with the caller's per-vector work. {@link #get(int,
   * float[])} unpacks directly into a caller-supplied array (no per-vector allocation).
   *
   * <p><b>Forward-only, single-consumer.</b> {@code get} must be called with non-decreasing indices
   * from a single thread — the intended pattern for streaming ingestion. To read only a slice (e.g.
   * one segment of a partitioned build), construct it with a {@code [firstVector, count)} range so
   * each segment streams just its own portion of the file.
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
    private final int firstVector; // absolute index of the first vector this reader serves
    private final int endVector; // absolute index just past the last vector this reader serves
    private final int vectorBytes;
    private final int chunkVectors;

    private final BlockingQueue<ByteBuffer> free = new ArrayBlockingQueue<>(2);
    private final BlockingQueue<Chunk> ready = new ArrayBlockingQueue<>(2);
    private final Thread reader;
    private volatile IOException readerError;

    private Chunk current; // consumer-owned; the chunk currently being served

    /** Reads the whole file from index 0. */
    PrefetchingFbinReader(Path path, int chunkSizeMB) throws IOException {
      this(path, 0, -1, chunkSizeMB);
    }

    /**
     * Reads the contiguous range {@code [firstVector, firstVector + count)}, or to end of file if
     * {@code count <= 0}. {@link #get} then serves absolute file indices within that range.
     */
    PrefetchingFbinReader(Path path, int firstVector, int count, int chunkSizeMB)
        throws IOException {
      this.channel = FileChannel.open(path, StandardOpenOption.READ);
      ByteBuffer header = ByteBuffer.allocate((int) HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
      readFully(header, 0);
      header.flip();
      int numVectors = header.getInt();
      this.dimension = header.getInt();
      this.vectorBytes = dimension * Float.BYTES;
      this.firstVector = firstVector;
      this.endVector =
          count > 0 ? (int) Math.min((long) firstVector + count, numVectors) : numVectors;

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
      return endVector - firstVector;
    }

    int dimension() {
      return dimension;
    }

    /** Reader thread: fill chunks front-to-back, blocking on a free buffer between chunks. */
    private void readLoop() {
      long next = firstVector;
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

    /** Fills {@code dst} with the vector at absolute {@code index} (no allocation). */
    void get(int index, float[] dst) throws IOException {
      if (index < firstVector || index >= endVector) {
        throw new IndexOutOfBoundsException(
            "Index " + index + " out of bounds [" + firstVector + ", " + endVector + ")");
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

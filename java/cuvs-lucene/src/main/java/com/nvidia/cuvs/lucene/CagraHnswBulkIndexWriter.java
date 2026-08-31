/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Semaphore;
import java.util.stream.Stream;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.misc.store.HardlinkCopyDirectoryWrapper;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * Builds a GPU-accelerated CAGRA/HNSW Lucene index from data the caller already has ready on
 * local disk, as few segments as possible, without going through a generically-configurable
 * {@link org.apache.lucene.codecs.Codec}.
 *
 * <p>This is <b>not</b> a general-purpose Lucene extension point: an instance owns a single
 * {@link IndexWriter} and its {@link IndexWriterConfig} so that the invariants the underlying GPU
 * writer requires (single unmerged segment, no index sort, exact vector count) are guaranteed by
 * this class rather than left to a caller to get right. If you need a general Lucene codec that
 * any indexing pipeline (including ones that don't control their own {@code IndexWriter}
 * lifecycle, e.g. Solr or Elasticsearch) can register, use {@link Lucene101AcceleratedHNSWCodec}
 * directly instead.
 *
 * <p><b>Two ways to use this class:</b>
 *
 * <ul>
 *   <li><b>Manual, single segment:</b> construct an instance directly, call {@link #addDocument}
 *       per document exactly like a plain {@link IndexWriter}, then {@link #close}. You own the
 *       loop and the {@link Document} you build (any fields, not just the vector).
 *   <li><b>One-shot, one or many segments:</b> {@link #indexFbin} / {@link #build(VectorSource,
 *       Config)} own the loop for you — they read vectors from a {@code .fbin} file or {@link
 *       VectorSource}, optionally split into {@code numSegments} partitions (sequential or
 *       overlapped), and combine the result. Since they build each row's {@link Document}
 *       internally, an optional {@link FieldCallback} lets you add extra fields to it.
 * </ul>
 *
 * <p><b>Scope: CAGRA_HNSW (GPU build, CPU search) only.</b> This class builds indexes for {@link
 * Lucene101AcceleratedHNSWCodec}. It does not support the GPU-search codec ({@code
 * CuVS2510GPUSearchCodec}) — that writer does not (yet) have the native flat-buffering
 * optimization this class relies on. Reading an existing index for CAGRA_SEARCH-style GPU search
 * is unaffected by this class either way; only bulk-building one is out of scope for now.
 */
public final class CagraHnswBulkIndexWriter implements Closeable {

  private static final int DEFAULT_CHUNK_SIZE_MB = 32;

  private final IndexWriter writer;
  private final int exactVectorCount;
  private int documentsAdded;
  private boolean closed;

  /**
   * Opens a single-segment, native-flat-buffered writer. {@code exactVectorCount} must equal the
   * number of {@link #addDocument} calls that will follow — the native buffer is pre-sized to it,
   * so {@link #close} rejects a mismatched count rather than let the underlying writer produce a
   * corrupt or incomplete segment.
   *
   * <p>{@code conf}'s {@code Analyzer}, {@code Similarity}, {@code InfoStream}, and {@code
   * OpenMode} are honored; an explicit {@code IndexSort} is rejected ({@link
   * IllegalArgumentException}), since native flat buffering does not support index-sorted
   * segments. Codec, merge policy, and flush thresholds are always owned by this class regardless
   * of what {@code conf} contains — {@link IndexWriterConfig} does not expose whether the caller
   * explicitly set those or left them at Lucene's defaults, so there is no reliable way to
   * validate-and-reject a caller-supplied value for them the way {@code IndexSort} can be; this
   * class simply never reads them from {@code conf}.
   *
   * <p>{@code config.targetDirectory()}, {@code config.numSegments()}, {@code
   * config.overlapped()}, and {@code config.pipelineDepth()} are not consulted here — {@code
   * directory} is passed explicitly, and this constructor always builds exactly one segment. Those
   * fields only matter to {@link #indexFbin} / {@link #build(VectorSource, Config)}.
   */
  public CagraHnswBulkIndexWriter(
      Directory directory, IndexWriterConfig conf, Config config, int exactVectorCount)
      throws Exception {
    Objects.requireNonNull(directory, "directory");
    Objects.requireNonNull(conf, "conf");
    Objects.requireNonNull(config, "config");
    if (exactVectorCount <= 0) {
      throw new IllegalArgumentException("exactVectorCount must be > 0, got " + exactVectorCount);
    }
    if (conf.getIndexSort() != null) {
      throw new IllegalArgumentException(
          "CagraHnswBulkIndexWriter does not support an index-sorted segment (native flat"
              + " buffering requires an unsorted single-segment build); leave"
              + " IndexWriterConfig.indexSort unset");
    }
    this.exactVectorCount = exactVectorCount;

    Codec codec = new Lucene101AcceleratedHNSWCodec(config.graphBuildParams(), exactVectorCount);
    IndexWriterConfig ownedConf =
        new IndexWriterConfig(conf.getAnalyzer())
            .setSimilarity(conf.getSimilarity())
            .setInfoStream(conf.getInfoStream())
            .setCodec(codec)
            .setUseCompoundFile(false)
            .setMaxBufferedDocs(Math.max(2, exactVectorCount + 1))
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH)
            .setMergePolicy(NoMergePolicy.INSTANCE)
            .setOpenMode(conf.getOpenMode());
    this.writer = new IndexWriter(directory, ownedConf);
  }

  /**
   * Adds one document, exactly like {@link IndexWriter#addDocument}. {@code doc} may contain any
   * fields — the vector field (matching {@link Config#fieldName()}) is routed into the native
   * flat buffer automatically by the underlying codec, the same way any {@link
   * KnnFloatVectorField} is for any Lucene codec; every other field is indexed normally.
   */
  public long addDocument(Iterable<? extends IndexableField> doc) throws IOException {
    if (closed) {
      throw new IllegalStateException("addDocument called after close()");
    }
    if (documentsAdded >= exactVectorCount) {
      throw new IllegalStateException(
          "addDocument called more than exactVectorCount (" + exactVectorCount + ") times");
    }
    long seqNo = writer.addDocument(doc);
    documentsAdded++;
    return seqNo;
  }

  /**
   * Runs the single native-buffered flush (this is where the GPU CAGRA build happens) and closes
   * the underlying writer. There is no separate {@code commit()}: unlike a plain {@link
   * IndexWriter}, only one flush is ever valid for this instance, so folding it into {@code
   * close()} removes any way to trigger it early with fewer than {@code exactVectorCount} vectors
   * added. Throws {@link IllegalStateException} if {@link #addDocument} was called fewer times
   * than {@code exactVectorCount} promised, discarding the buffered documents rather than
   * flushing an under-filled segment.
   */
  @Override
  public void close() throws IOException {
    if (closed) {
      return;
    }
    closed = true;
    if (documentsAdded != exactVectorCount) {
      IllegalStateException mismatch =
          new IllegalStateException(
              "expected " + exactVectorCount + " documents, got " + documentsAdded);
      // The native host matrix is preallocated for exactly exactVectorCount rows, so an
      // under-filled buffer can only be discarded, never flushed. rollback() discards it and
      // closes the IndexWriter, and is also what frees that preallocated memory: it aborts the
      // indexing chain, which closes the vectors writer. Leaving without it leaks the buffer,
      // which lives in a shared Arena and is never reclaimed by the GC.
      try {
        writer.rollback();
      } catch (Throwable t) {
        mismatch.addSuppressed(t);
      }
      throw mismatch;
    }
    try (writer) {
      writer.commit();
    }
  }

  /**
   * Discards everything added so far without producing a segment: no GPU build runs, nothing is
   * committed, and the preallocated native buffer is released (see {@link #close()} for why that
   * release is what matters). Unlike {@link #close()} this does not care how many documents were
   * added, so it is the cleanup to use on a failure path -- a partially filled buffer is not
   * worth building, and reporting its count would only bury the failure that caused it.
   * Idempotent, and a no-op once {@link #close()} has run.
   */
  void abort() throws IOException {
    if (closed) {
      return;
    }
    closed = true;
    writer.rollback();
  }

  /**
   * Callback invoked once per row by {@link #indexFbin} / {@link #build(VectorSource, Config,
   * FieldCallback)}, right after the id and vector fields have been added to {@code document} and
   * right before it is added to the index — lets the caller attach additional fields (metadata)
   * per vector. Not used by the manual, direct-instance API, where the caller already builds the
   * whole {@link Document} themselves.
   */
  @FunctionalInterface
  public interface FieldCallback {
    void addFields(Document document, int id) throws IOException;
  }

  /**
   * Builds an index from {@code source} into {@code config.targetDirectory()}, partitioned into
   * {@code config.numSegments()} contiguous slices as {@code source} is consumed front-to-back.
   *
   * <p>{@code config.overlapped()} is not supported here: a single {@link VectorSource} is
   * forward-only/single-consumer (see its contract) and so cannot be read concurrently by
   * multiple slice builders. Use {@link #indexFbin} for the overlapped pipeline, which knows how
   * to open independent, per-slice sources over the same underlying file.
   */
  public static void build(VectorSource source, Config config) throws Exception {
    build(source, config, null);
  }

  /** As {@link #build(VectorSource, Config)}, with a {@link FieldCallback} for extra fields. */
  public static void build(VectorSource source, Config config, FieldCallback callback)
      throws Exception {
    Objects.requireNonNull(source, "source");
    Objects.requireNonNull(config, "config");
    if (config.overlapped()) {
      throw new IllegalArgumentException(
          "Config.overlapped() is not supported by build(VectorSource, Config): a VectorSource is"
              + " forward-only and cannot be read by multiple concurrent slice builders. Use"
              + " indexFbin(...) for the overlapped multi-segment build.");
    }
    if (source.dimensions() != config.dimensions()) {
      throw new IllegalArgumentException(
          "source.dimensions() ("
              + source.dimensions()
              + ") does not match Config.dimensions() ("
              + config.dimensions()
              + ")");
    }
    List<int[]> slices = sliceEvenly(source.size(), config.numSegments());
    buildSequential(source, config, callback, slices);
  }

  /**
   * Convenience entry point: builds an index directly from a {@code .fbin} file ({@code
   * [num_vectors int32][dim int32]} header, then contiguous little-endian float32 rows), using
   * {@link FbinVectorSource} with {@link #DEFAULT_CHUNK_SIZE_MB}-sized prefetched chunks.
   */
  public static void indexFbin(Path fbinPath, Config config) throws Exception {
    indexFbin(fbinPath, config, null, DEFAULT_CHUNK_SIZE_MB);
  }

  /** As {@link #indexFbin(Path, Config)}, with a {@link FieldCallback} for extra fields. */
  public static void indexFbin(Path fbinPath, Config config, FieldCallback callback)
      throws Exception {
    indexFbin(fbinPath, config, callback, DEFAULT_CHUNK_SIZE_MB);
  }

  /** As {@link #indexFbin(Path, Config)}, with an explicit prefetch chunk size. */
  public static void indexFbin(Path fbinPath, Config config, int chunkSizeMB) throws Exception {
    indexFbin(fbinPath, config, null, chunkSizeMB);
  }

  /**
   * As {@link #indexFbin(Path, Config)}, with an explicit prefetch chunk size and {@link
   * FieldCallback}. When {@code config.numSegments() > 1} and {@code config.overlapped()}, builds
   * up to {@code config.pipelineDepth()} segments concurrently, each over its own slice of {@code
   * fbinPath}, then combines them by hardlink; otherwise builds sequentially over one shared
   * reader.
   */
  public static void indexFbin(
      Path fbinPath, Config config, FieldCallback callback, int chunkSizeMB) throws Exception {
    Objects.requireNonNull(fbinPath, "fbinPath");
    Objects.requireNonNull(config, "config");
    int total;
    int dim;
    try (FbinVectorSource probe = new FbinVectorSource(fbinPath, 1)) {
      total = probe.size();
      dim = probe.dimensions();
    }
    if (dim != config.dimensions()) {
      throw new IllegalArgumentException(
          "fbinPath dimension ("
              + dim
              + ") does not match Config.dimensions() ("
              + config.dimensions()
              + ")");
    }
    List<int[]> slices = sliceEvenly(total, config.numSegments());
    if (config.overlapped() && slices.size() > 1) {
      buildOverlapped(fbinPath, config, callback, chunkSizeMB, slices, dim);
    } else {
      try (FbinVectorSource source = new FbinVectorSource(fbinPath, chunkSizeMB)) {
        buildSequential(source, config, callback, slices);
      }
    }
  }

  /**
   * Sequential partitioned build: {@code source} is streamed front-to-back across all slices,
   * each slice built as a single native-flat segment appended to the same directory (first slice
   * {@code CREATE}, later slices {@code APPEND}). Peak host memory is one slice's native buffer.
   */
  private static void buildSequential(
      VectorSource source, Config config, FieldCallback callback, List<int[]> slices)
      throws Exception {
    float[] scratch = new float[config.dimensions()];
    try (Directory dir = FSDirectory.open(config.targetDirectory())) {
      for (int p = 0; p < slices.size(); p++) {
        int[] slice = slices.get(p);
        buildSegment(
            dir, source, scratch, config, callback, slice[0], slice[0], slice[1], p == 0, null);
      }
    }
  }

  /**
   * Overlapped partitioned build: a bounded pool builds up to {@code config.pipelineDepth()}
   * segments at once, each with its OWN {@link FbinVectorSource} over just its slice, so a
   * segment's ingest overlaps a prior segment's GPU commit. The GPU build itself is serialized on
   * a single permit. The finished per-segment indexes are combined into {@code
   * config.targetDirectory()} by hardlinking their files (no bulk copy of the vector data).
   */
  private static void buildOverlapped(
      Path fbinPath,
      Config config,
      FieldCallback callback,
      int chunkSizeMB,
      List<int[]> slices,
      int dim)
      throws Exception {
    Path targetDir = config.targetDirectory();
    int depth = Math.min(slices.size(), config.pipelineDepth());
    List<Path> segDirs = new ArrayList<>();
    for (int p = 0; p < slices.size(); p++) {
      segDirs.add(targetDir.resolveSibling(targetDir.getFileName() + "_p" + p));
    }
    for (Path segDir : segDirs) {
      deleteRecursivelyQuietly(segDir);
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
                  try (FbinVectorSource source =
                          new FbinVectorSource(fbinPath, slice[0], slice[1], chunkSizeMB);
                      Directory d = FSDirectory.open(segDir)) {
                    // createNew=true: each segment is a fresh single-segment index in its own
                    // dir; sourceStart=0 since this source is already windowed to the slice.
                    buildSegment(
                        d, source, scratch, config, callback, 0, slice[0], slice[1], true,
                        gpuPermit);
                  }
                  return null;
                }));
      }
      pool.shutdown();
      try {
        for (Future<?> f : futures) {
          f.get(); // propagate any build failure
        }
      } catch (Exception e) {
        throw new IOException("Overlapped bulk index build failed", e);
      } finally {
        pool.shutdownNow();
      }
      combineByHardlink(targetDir, segDirs);
    } finally {
      for (Path segDir : segDirs) {
        deleteRecursivelyQuietly(segDir);
      }
    }
  }

  /**
   * Builds one segment from {@code size} vectors: {@code source.get(sourceStart + i, ...)} for
   * {@code i} in {@code [0, size)}, labelled with ids {@code idStart + i} (the vectors' absolute
   * position in the overall build, regardless of whether {@code source} itself is windowed). Opens
   * a {@link CagraHnswBulkIndexWriter} sized to {@code size} and drives it exactly like the manual
   * API. When {@code gpuPermit} is non-null the close (which runs the GPU CAGRA build) is
   * serialized on it while other segments' host-side ingest may proceed.
   */
  private static void buildSegment(
      Directory dir,
      VectorSource source,
      float[] scratch,
      Config config,
      FieldCallback callback,
      int sourceStart,
      int idStart,
      int size,
      boolean createNew,
      Semaphore gpuPermit)
      throws Exception {
    IndexWriterConfig conf =
        new IndexWriterConfig()
            .setOpenMode(
                createNew ? IndexWriterConfig.OpenMode.CREATE : IndexWriterConfig.OpenMode.APPEND);
    CagraHnswBulkIndexWriter writer = new CagraHnswBulkIndexWriter(dir, conf, config, size);
    try {
      for (int i = 0; i < size; i++) {
        source.get(sourceStart + i, scratch);
        int id = idStart + i;
        Document doc = new Document();
        if (config.idFieldName() != null) {
          doc.add(new StringField(config.idFieldName(), Integer.toString(id), Field.Store.YES));
        }
        doc.add(new KnnFloatVectorField(config.fieldName(), scratch, config.similarity()));
        if (callback != null) {
          callback.addFields(doc, id);
        }
        writer.addDocument(doc); // copies the vector -> 'scratch' is safe to reuse next iteration
      }
      // The single flush inside close() is where the GPU CAGRA build runs; serialize it if asked.
      if (gpuPermit != null) {
        gpuPermit.acquire();
        try {
          writer.close();
        } finally {
          gpuPermit.release();
        }
      } else {
        writer.close();
      }
    } catch (Throwable t) {
      // Cleanup must not become the reported failure. abort() discards the partially filled
      // buffer rather than trying to build it, and anything it throws on the way out is attached
      // to the original exception instead of replacing it. If the close() above is what failed,
      // this is a no-op -- that writer has already released itself.
      try {
        writer.abort();
      } catch (Throwable cleanupFailure) {
        t.addSuppressed(cleanupFailure);
      }
      throw t;
    }
  }

  /**
   * Combines the per-segment indexes into {@code targetDir} by hardlinking their files (same
   * filesystem) rather than copying the vector data. {@link HardlinkCopyDirectoryWrapper} falls
   * back to a byte copy automatically if the segment dirs and the final dir are on different
   * filesystems.
   */
  private static void combineByHardlink(Path targetDir, List<Path> segDirs) throws IOException {
    Directory[] sources = new Directory[segDirs.size()];
    try {
      for (int i = 0; i < segDirs.size(); i++) {
        sources[i] = FSDirectory.open(segDirs.get(i));
      }
      IndexWriterConfig iwc =
          new IndexWriterConfig().setMergePolicy(NoMergePolicy.INSTANCE); // keep segments separate
      try (Directory target = new HardlinkCopyDirectoryWrapper(FSDirectory.open(targetDir));
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

  private static void deleteRecursivelyQuietly(Path path) {
    if (!Files.exists(path)) {
      return;
    }
    try (Stream<Path> walk = Files.walk(path)) {
      walk.sorted(Comparator.reverseOrder())
          .forEach(
              p -> {
                try {
                  Files.deleteIfExists(p);
                } catch (IOException ignored) {
                  // best-effort cleanup of a temp per-segment dir
                }
              });
    } catch (IOException ignored) {
      // best-effort cleanup of a temp per-segment dir
    }
  }

  /** Immutable configuration for {@link CagraHnswBulkIndexWriter}. */
  public static final class Config {
    private final String fieldName;
    private final int dimensions;
    private final VectorSimilarityFunction similarity;
    private final String idFieldName;
    private final AcceleratedHNSWParams graphBuildParams;
    private final Path targetDirectory;
    private final int numSegments;
    private final boolean overlapped;
    private final int pipelineDepth;

    private Config(Builder b) {
      this.fieldName = b.fieldName;
      this.dimensions = b.dimensions;
      this.similarity = b.similarity;
      this.idFieldName = b.idFieldName;
      this.graphBuildParams = b.graphBuildParams;
      this.targetDirectory = b.targetDirectory;
      this.numSegments = b.numSegments;
      this.overlapped = b.overlapped;
      this.pipelineDepth = b.pipelineDepth;
    }

    public String fieldName() {
      return fieldName;
    }

    public int dimensions() {
      return dimensions;
    }

    public VectorSimilarityFunction similarity() {
      return similarity;
    }

    public String idFieldName() {
      return idFieldName;
    }

    public AcceleratedHNSWParams graphBuildParams() {
      return graphBuildParams;
    }

    /** Only consulted by {@link #indexFbin} / {@link #build(VectorSource, Config)}. */
    public Path targetDirectory() {
      return targetDirectory;
    }

    /** Only consulted by {@link #indexFbin} / {@link #build(VectorSource, Config)}. */
    public int numSegments() {
      return numSegments;
    }

    /** Only consulted by {@link #indexFbin}. */
    public boolean overlapped() {
      return overlapped;
    }

    /** Only consulted by {@link #indexFbin}. */
    public int pipelineDepth() {
      return pipelineDepth;
    }

    public static Builder builder() {
      return new Builder();
    }

    /** Builder for {@link Config}. */
    public static final class Builder {
      private String fieldName = "vector";
      private int dimensions = -1;
      private VectorSimilarityFunction similarity = VectorSimilarityFunction.EUCLIDEAN;
      private String idFieldName = "id";
      private AcceleratedHNSWParams graphBuildParams;
      private Path targetDirectory;
      private int numSegments = 1;
      private boolean overlapped = false;
      private int pipelineDepth = 2;

      /** Sets the vector field name and dimensionality; required. */
      public Builder field(String fieldName, int dimensions, VectorSimilarityFunction similarity) {
        this.fieldName = Objects.requireNonNull(fieldName, "fieldName");
        this.dimensions = dimensions;
        this.similarity = Objects.requireNonNull(similarity, "similarity");
        return this;
      }

      /**
       * Sets the stored id field name (holding each document's absolute position in the build, as
       * a string), added automatically by {@link #indexFbin} / {@link #build(VectorSource,
       * Config)} before the {@link FieldCallback} runs. Defaults to {@code "id"}; pass {@code
       * null} to disable. Not used by the manual, direct-instance API.
       */
      public Builder idField(String idFieldName) {
        this.idFieldName = idFieldName;
        return this;
      }

      /** Sets the CAGRA/HNSW graph-build parameters; required. */
      public Builder graphBuild(AcceleratedHNSWParams graphBuildParams) {
        this.graphBuildParams = Objects.requireNonNull(graphBuildParams, "graphBuildParams");
        return this;
      }

      /** Sets the directory the final index is written to; required for {@link #indexFbin}/{@link #build}. */
      public Builder targetDirectory(Path targetDirectory) {
        this.targetDirectory = Objects.requireNonNull(targetDirectory, "targetDirectory");
        return this;
      }

      /**
       * Splits the build into {@code numSegments} contiguous slices, each a single native-flat
       * segment. Peak host memory scales as {@code 1/numSegments}; the GPU build itself is always
       * serialized across slices regardless of this setting. Default 1 (single segment).
       *
       * @param overlapped when {@code numSegments > 1} and building via {@link #indexFbin},
       *     builds up to {@link #pipelineDepth} slices concurrently (ingest of one overlapping the
       *     GPU commit of another) instead of strictly sequentially. Ignored by {@link
       *     #build(VectorSource, Config)}, which always builds sequentially — see that method's
       *     Javadoc.
       */
      public Builder segments(int numSegments, boolean overlapped) {
        if (numSegments < 1) {
          throw new IllegalArgumentException("numSegments must be >= 1, got " + numSegments);
        }
        this.numSegments = numSegments;
        this.overlapped = overlapped;
        return this;
      }

      /** Max segments built concurrently in overlap mode; peak host memory is this many slice buffers. */
      public Builder pipelineDepth(int pipelineDepth) {
        if (pipelineDepth < 1) {
          throw new IllegalArgumentException("pipelineDepth must be >= 1, got " + pipelineDepth);
        }
        this.pipelineDepth = pipelineDepth;
        return this;
      }

      public Config build() {
        if (dimensions <= 0) {
          throw new IllegalStateException(
              "field(...) must be called with a positive dimension count");
        }
        Objects.requireNonNull(graphBuildParams, "graphBuild(...) must be called");
        return new Config(this);
      }
    }
  }
}

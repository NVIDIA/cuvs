/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import java.io.Closeable;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
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
import org.apache.lucene.index.VectorEncoding;
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
 * writer requires (one controlled flush per segment, no index sort or merges, exact vector count)
 * are guaranteed by this class rather than left to a caller to get right. If you need a general
 * Lucene codec that any indexing pipeline (including ones that don't control their own {@code
 * IndexWriter} lifecycle, e.g. Solr or Elasticsearch) can register, use {@link
 * Lucene101AcceleratedHNSWCodec} directly instead.
 *
 * <p><b>Two ways to use this class:</b>
 *
 * <ul>
 *   <li><b>Manual, single segment:</b> construct an instance directly, call {@link #addDocument}
 *       per document exactly like a plain {@link IndexWriter}, then {@link #close}. You own the
 *       loop and the {@link Document} you build (any fields, not just the vector).
 *   <li><b>One-shot, one or many segments:</b> {@link #indexFbin}, {@link #indexMappedFbin}, {@link
 *       #indexImmutableFbin}, and {@link #build(VectorSource, Config)} own the loop for you. They
 *       optionally split the input into {@code numSegments} partitions, build one independent
 *       graph per partition, and combine them without merging. Since they build each row's {@link
 *       Document} internally, an optional {@link FieldCallback} lets you add extra fields to it.
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
  private final Config config;
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
   * fields only matter to the one-shot build methods.
   */
  public CagraHnswBulkIndexWriter(
      Directory directory, IndexWriterConfig conf, Config config, int exactVectorCount)
      throws Exception {
    this(
        directory,
        conf,
        config,
        BulkIndexingContext.nativeBuffered(exactVectorCount, config.metrics()));
  }

  private CagraHnswBulkIndexWriter(
      Directory directory, IndexWriterConfig conf, Config config, BulkIndexingContext bulkContext)
      throws Exception {
    Objects.requireNonNull(directory, "directory");
    Objects.requireNonNull(conf, "conf");
    Objects.requireNonNull(config, "config");
    Objects.requireNonNull(bulkContext, "bulkContext");
    int exactVectorCount = bulkContext.exactVectorCount();
    if (exactVectorCount <= 0) {
      throw new IllegalArgumentException("exactVectorCount must be > 0, got " + exactVectorCount);
    }
    if (conf.getIndexSort() != null) {
      throw new IllegalArgumentException(
          "CagraHnswBulkIndexWriter does not support an index-sorted segment (native flat"
              + " buffering requires an unsorted segment build); leave"
              + " IndexWriterConfig.indexSort unset");
    }
    this.config = config;
    this.exactVectorCount = exactVectorCount;

    Codec codec = new Lucene101AcceleratedHNSWCodec(config.graphBuildParams(), bulkContext);
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
   * Adds one document, exactly like {@link IndexWriter#addDocument}. {@code doc} must contain
   * exactly one vector field whose name, dimension, and similarity match {@code config}; that field
   * is routed into the native flat buffer automatically by the underlying codec. Any number of
   * non-vector fields may also be present and are indexed normally.
   */
  public long addDocument(Iterable<? extends IndexableField> doc) throws IOException {
    if (closed) {
      throw new IllegalStateException("addDocument called after close()");
    }
    if (documentsAdded >= exactVectorCount) {
      throw new IllegalStateException(
          "addDocument called more than exactVectorCount (" + exactVectorCount + ") times");
    }
    List<IndexableField> fields = validateAndSnapshotDocument(doc);
    long seqNo = writer.addDocument(fields);
    documentsAdded++;
    return seqNo;
  }

  private List<IndexableField> validateAndSnapshotDocument(
      Iterable<? extends IndexableField> document) {
    Objects.requireNonNull(document, "doc");
    List<IndexableField> fields = new ArrayList<>();
    int vectorFields = 0;
    for (IndexableField field : document) {
      fields.add(field);
      var fieldType = field.fieldType();
      if (fieldType.vectorDimension() == 0) {
        continue;
      }
      vectorFields++;
      if (!field.name().equals(config.fieldName())) {
        throw new IllegalArgumentException(
            "Vector field name \""
                + field.name()
                + "\" does not match configured field \""
                + config.fieldName()
                + "\"");
      }
      if (fieldType.vectorDimension() != config.dimensions()) {
        throw new IllegalArgumentException(
            "Vector field dimension "
                + fieldType.vectorDimension()
                + " does not match configured dimension "
                + config.dimensions());
      }
      if (fieldType.vectorEncoding() != VectorEncoding.FLOAT32) {
        throw new IllegalArgumentException("CAGRA/HNSW bulk indexing requires FLOAT32 vectors");
      }
      if (fieldType.vectorSimilarityFunction() != config.similarity()) {
        throw new IllegalArgumentException(
            "Vector field similarity "
                + fieldType.vectorSimilarityFunction()
                + " does not match configured similarity "
                + config.similarity());
      }
    }
    if (vectorFields != 1) {
      throw new IllegalArgumentException(
          "Each bulk-indexed document must contain exactly one vector field; found "
              + vectorFields);
    }
    return fields;
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
    try (Closeable writerCloser = this::closeUnderlyingWriter) {
      long commitStartedAt = CagraHnswBuildMetrics.start();
      try {
        writer.commit();
      } finally {
        config.metrics().stop("bulk writer commit wall [CPU+GPU+DISK]", commitStartedAt);
      }
    }
  }

  private void closeUnderlyingWriter() throws IOException {
    long closeStartedAt = CagraHnswBuildMetrics.start();
    try {
      writer.close();
    } finally {
      config.metrics().stop("bulk writer close [DISK]", closeStartedAt);
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
   * Callback invoked once per row by the one-shot build methods, right after the id and vector
   * fields have been added to {@code document} and right before it is added to the index. This lets
   * the caller attach additional fields (metadata) per vector. Overlapped builds may invoke the
   * callback concurrently, so callback implementations must be thread-safe. Not used by the
   * manual, direct-instance API, where the caller already builds the whole {@link Document}
   * themselves.
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
    FbinFileMetadata metadata = FbinFileMetadata.read(fbinPath);
    int total = metadata.rows();
    int dim = metadata.dimensions();
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
   * Builds a normal, self-contained Lucene index while CAGRA and the flat-vector writer consume a
   * shared read-only mapping of {@code fbinPath}. This avoids decoding and copying every row during
   * document ingest while preserving a conventional {@code .vec} file in the finished index.
   */
  public static void indexMappedFbin(Path fbinPath, Config config) throws Exception {
    indexMappedFbin(fbinPath, config, null);
  }

  /** As {@link #indexMappedFbin(Path, Config)}, with a callback for non-vector fields. */
  public static void indexMappedFbin(Path fbinPath, Config config, FieldCallback callback)
      throws Exception {
    Objects.requireNonNull(fbinPath, "fbinPath");
    Objects.requireNonNull(config, "config");
    requireMappedBuildTarget(config);
    if (config.numSegments() > 1) {
      FbinFileMetadata metadata = FbinFileMetadata.read(fbinPath);
      validateMappedShape(metadata.rows(), metadata.dimensions(), config);
      List<int[]> slices = sliceEvenly(metadata.rows(), config.numSegments());
      buildBorrowedSegments(
          config,
          slices,
          (segmentDirectory, firstRow, rowCount, gpuPermit) -> {
            try (MappedFbinDataset mapped = MappedFbinDataset.map(fbinPath, firstRow, rowCount)) {
              buildBorrowedSegment(
                  segmentDirectory,
                  config,
                  callback,
                  BulkIndexingContext.mapped(mapped.dataset(), config.metrics()),
                  firstRow,
                  gpuPermit);
            }
          },
          null,
          null);
      return;
    }
    try (MappedFbinDataset mapped = MappedFbinDataset.map(fbinPath)) {
      validateMappedShape(mapped.rows(), mapped.dimensions(), config);
      buildBorrowedSegment(
          config.targetDirectory(),
          config,
          callback,
          BulkIndexingContext.mapped(mapped.dataset(), config.metrics()));
    }
  }

  /**
   * Builds a non-self-contained Lucene index whose exact-vector scoring reads from the immutable
   * FBIN represented by {@code source}. The caller must retain {@code source} through every reader
   * lifetime and replicate the referenced FBIN together with the Lucene directory.
   */
  public static void indexImmutableFbin(
      ExternalFbinFileRegistry.Registration source, Config config, ExternalFbinOptions options)
      throws Exception {
    indexImmutableFbin(source, config, options, null);
  }

  /** As {@link #indexImmutableFbin(ExternalFbinFileRegistry.Registration, Config,
   * ExternalFbinOptions)}, with a callback for non-vector fields. */
  public static void indexImmutableFbin(
      ExternalFbinFileRegistry.Registration source,
      Config config,
      ExternalFbinOptions options,
      FieldCallback callback)
      throws Exception {
    Objects.requireNonNull(source, "source");
    Objects.requireNonNull(config, "config");
    Objects.requireNonNull(options, "options");
    requireMappedBuildTarget(config);
    FbinFileMetadata metadata = FbinFileMetadata.read(source.path());
    int rows = metadata.rows();
    int dimensions = metadata.dimensions();
    validateMappedShape(rows, dimensions, config);
    if (config.numSegments() > 1) {
      List<int[]> slices = sliceEvenly(rows, config.numSegments());
      ExternalFbinOptions trustedOptions =
          new ExternalFbinOptions(ExternalFbinBuildValidation.TRUSTED_IMMUTABLE, 0L);
      buildBorrowedSegments(
          config,
          slices,
          (segmentDirectory, firstRow, rowCount, gpuPermit) -> {
            try (ImmutableExternalFbinDataset dataset = source.map(firstRow, rowCount)) {
              buildBorrowedSegment(
                  segmentDirectory,
                  config,
                  callback,
                  BulkIndexingContext.external(dataset, trustedOptions, config.metrics()),
                  firstRow,
                  gpuPermit);
            }
          },
          source.reference(0, rows),
          options);
      return;
    }
    try (ImmutableExternalFbinDataset dataset = source.map(0, rows)) {
      buildBorrowedSegment(
          config.targetDirectory(),
          config,
          callback,
          BulkIndexingContext.external(dataset, options, config.metrics()));
    }
  }

  private static void requireMappedBuildTarget(Config config) {
    if (config.targetDirectory() == null) {
      throw new IllegalArgumentException("targetDirectory is required for a one-shot bulk build");
    }
  }

  private static void validateMappedShape(int rows, int dimensions, Config config) {
    if (rows <= 0 || dimensions != config.dimensions()) {
      throw new IllegalArgumentException(
          "FBIN shape "
              + rows
              + " x "
              + dimensions
              + " does not match configured dimension "
              + config.dimensions());
    }
  }

  private static void buildBorrowedSegment(
      Path targetDirectory, Config config, FieldCallback callback, BulkIndexingContext context)
      throws Exception {
    buildBorrowedSegment(targetDirectory, config, callback, context, 0, null);
  }

  private static void buildBorrowedSegment(
      Path targetDirectory,
      Config config,
      FieldCallback callback,
      BulkIndexingContext context,
      int idStart,
      Semaphore gpuPermit)
      throws Exception {
    long ingestStartedAt = CagraHnswBuildMetrics.start();
    try (Directory directory = FSDirectory.open(targetDirectory)) {
      IndexWriterConfig writerConfig =
          new IndexWriterConfig().setOpenMode(IndexWriterConfig.OpenMode.CREATE);
      CagraHnswBulkIndexWriter writer =
          new CagraHnswBulkIndexWriter(directory, writerConfig, config, context);
      try {
        addBorrowedDocuments(writer, config, callback, idStart, context.exactVectorCount());
        config.metrics().stop("borrowed document ingest [CPU]", ingestStartedAt);
        boolean permitAcquired = false;
        try {
          if (gpuPermit != null) {
            gpuPermit.acquire();
            permitAcquired = true;
          }
          writer.close();
        } finally {
          if (permitAcquired) {
            gpuPermit.release();
          }
        }
      } catch (Throwable failure) {
        try {
          writer.abort();
        } catch (Throwable cleanupFailure) {
          failure.addSuppressed(cleanupFailure);
        }
        throw failure;
      }
    }
  }

  private static void addBorrowedDocuments(
      CagraHnswBulkIndexWriter writer,
      Config config,
      FieldCallback callback,
      int idStart,
      int count)
      throws IOException {
    float[] placeholder = new float[config.dimensions()];
    if (callback == null) {
      Document document = new Document();
      StringField idField =
          config.idFieldName() == null
              ? null
              : new StringField(config.idFieldName(), "0", Field.Store.YES);
      if (idField != null) {
        document.add(idField);
      }
      document.add(new KnnFloatVectorField(config.fieldName(), placeholder, config.similarity()));
      for (int localId = 0; localId < count; localId++) {
        int id = Math.addExact(idStart, localId);
        if (idField != null) {
          idField.setStringValue(Integer.toString(id));
        }
        writer.addDocument(document);
      }
      return;
    }

    for (int localId = 0; localId < count; localId++) {
      int id = Math.addExact(idStart, localId);
      Document document = new Document();
      if (config.idFieldName() != null) {
        document.add(new StringField(config.idFieldName(), Integer.toString(id), Field.Store.YES));
      }
      document.add(new KnnFloatVectorField(config.fieldName(), placeholder, config.similarity()));
      callback.addFields(document, id);
      writer.addDocument(document);
    }
  }

  @FunctionalInterface
  private interface BorrowedSegmentBuilder {
    void build(Path segmentDirectory, int firstRow, int rowCount, Semaphore gpuPermit)
        throws Exception;
  }

  private static void buildBorrowedSegments(
      Config config,
      List<int[]> slices,
      BorrowedSegmentBuilder segmentBuilder,
      ExternalFbinReference validationReference,
      ExternalFbinOptions validationOptions)
      throws Exception {
    Path target = config.targetDirectory().toAbsolutePath();
    Path parent = target.getParent();
    Files.createDirectories(parent);
    Path temporaryRoot =
        Files.createTempDirectory(parent, target.getFileName().toString() + ".bulk-");
    List<Path> segmentDirectories = new ArrayList<>(slices.size());
    for (int segment = 0; segment < slices.size(); segment++) {
      segmentDirectories.add(temporaryRoot.resolve("segment-" + segment));
    }

    try {
      if (validationReference == null
          || validationOptions.validation() == ExternalFbinBuildValidation.TRUSTED_IMMUTABLE) {
        executeBorrowedSegmentBuilds(config, slices, segmentDirectories, segmentBuilder);
      } else {
        long overlapStartedAt = CagraHnswBuildMetrics.start();
        try {
          ExternalFbinScanCoordinator coordinator =
              ExternalFbinScanCoordinator.start(
                  validationReference,
                  validationOptions.validation(),
                  validationOptions.scanHeadStartBytes(),
                  config.metrics());
          coordinator.runAfterHeadStart(
              () ->
                  executeBorrowedSegmentBuilds(config, slices, segmentDirectories, segmentBuilder));
        } finally {
          config.metrics().stop("external reference overlap wall [GPU+DISK]", overlapStartedAt);
        }
      }

      long combineStartedAt = CagraHnswBuildMetrics.start();
      try {
        combineByHardlink(target, segmentDirectories);
      } finally {
        config.metrics().stop("segment combine [DISK]", combineStartedAt);
      }
    } finally {
      deleteRecursivelyQuietly(temporaryRoot);
    }
  }

  private static void executeBorrowedSegmentBuilds(
      Config config,
      List<int[]> slices,
      List<Path> segmentDirectories,
      BorrowedSegmentBuilder segmentBuilder)
      throws Exception {
    if (!config.overlapped()) {
      for (int segment = 0; segment < slices.size(); segment++) {
        int[] slice = slices.get(segment);
        segmentBuilder.build(segmentDirectories.get(segment), slice[0], slice[1], null);
      }
      return;
    }

    int depth = Math.min(slices.size(), config.pipelineDepth());
    Semaphore gpuPermit = new Semaphore(1);
    ExecutorService pool = Executors.newFixedThreadPool(depth);
    List<Future<?>> futures = new ArrayList<>(slices.size());
    try {
      for (int segment = 0; segment < slices.size(); segment++) {
        int[] slice = slices.get(segment);
        Path segmentDirectory = segmentDirectories.get(segment);
        futures.add(
            pool.submit(
                () -> {
                  segmentBuilder.build(segmentDirectory, slice[0], slice[1], gpuPermit);
                  return null;
                }));
      }
      for (Future<?> future : futures) {
        try {
          future.get();
        } catch (InterruptedException interrupted) {
          Thread.currentThread().interrupt();
          InterruptedIOException failure =
              new InterruptedIOException("Interrupted while awaiting a bulk segment build");
          failure.initCause(interrupted);
          throw failure;
        } catch (ExecutionException failure) {
          rethrowSegmentFailure(failure.getCause());
        }
      }
    } finally {
      for (Future<?> future : futures) {
        future.cancel(true);
      }
      pool.shutdownNow();
      pool.close();
    }
  }

  private static void rethrowSegmentFailure(Throwable failure) throws Exception {
    if (failure instanceof Exception exception) {
      throw exception;
    }
    if (failure instanceof Error error) {
      throw error;
    }
    throw new RuntimeException("Unexpected segment-build failure", failure);
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
          new IndexWriterConfig()
              .setMergePolicy(NoMergePolicy.INSTANCE)
              .setOpenMode(IndexWriterConfig.OpenMode.CREATE); // keep only imported segments
      try (Directory target = new HardlinkCopyDirectoryWrapper(FSDirectory.open(targetDir))) {
        IndexWriter combiner = new IndexWriter(target, iwc);
        try {
          combiner.addIndexes(sources);
          combiner.close();
        } catch (Throwable failure) {
          try {
            combiner.rollback();
          } catch (Throwable cleanupFailure) {
            failure.addSuppressed(cleanupFailure);
          }
          throw Utils.handleThrowable(failure);
        }
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
    private final CagraHnswBuildMetrics metrics;

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
      this.metrics = b.metrics;
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

    /** Only consulted by the one-shot build methods. */
    public Path targetDirectory() {
      return targetDirectory;
    }

    /** Only consulted by the one-shot build methods. */
    public int numSegments() {
      return numSegments;
    }

    /** Only consulted by one-shot methods that can open independent input slices. */
    public boolean overlapped() {
      return overlapped;
    }

    /** Only consulted by one-shot methods that can open independent input slices. */
    public int pipelineDepth() {
      return pipelineDepth;
    }

    /** Per-build metrics accumulator shared by all segments in this configuration. */
    public CagraHnswBuildMetrics metrics() {
      return metrics;
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
      private CagraHnswBuildMetrics metrics = new CagraHnswBuildMetrics();

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
       * @param overlapped when {@code numSegments > 1}, builds up to {@link #pipelineDepth} slices
       *     concurrently instead of strictly sequentially. Supported by the FBIN entry points and
       *     ignored by {@link #build(VectorSource, Config)}, whose caller-owned source is
       *     single-consumer.
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

      /** Supplies a caller-owned metrics accumulator for this build. */
      public Builder metrics(CagraHnswBuildMetrics metrics) {
        this.metrics = Objects.requireNonNull(metrics, "metrics");
        return this;
      }

      public Config build() {
        if (dimensions <= 0) {
          throw new IllegalStateException(
              "field(...) must be called with a positive dimension count");
        }
        Objects.requireNonNull(graphBuildParams, "graphBuild(...) must be called");
        CuvsDistanceType expectedMetric =
            switch (similarity) {
              case EUCLIDEAN -> CuvsDistanceType.L2Expanded;
              case DOT_PRODUCT, MAXIMUM_INNER_PRODUCT -> CuvsDistanceType.InnerProduct;
              case COSINE -> CuvsDistanceType.CosineExpanded;
            };
        if (graphBuildParams.getCuvsDistanceType() != expectedMetric) {
          throw new IllegalArgumentException(
              "Graph metric "
                  + graphBuildParams.getCuvsDistanceType()
                  + " does not match field similarity "
                  + similarity
                  + " (expected "
                  + expectedMetric
                  + ")");
        }
        return new Config(this);
      }
    }
  }
}

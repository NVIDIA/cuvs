/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.generateDataset;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;
import java.util.UUID;
import org.apache.commons.io.FileUtils;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/**
 * Negative-path coverage for the guard rails in {@code NativeFlatBufferedHNSWVectorsWriter} around
 * {@code numInputVectors} (native flat buffering): a count mismatch, an index-sorted segment, a
 * merge attempt, and that a count mismatch on one field doesn't leave another field's native
 * buffer unreleased on close.
 *
 * <p>Positive-path coverage (does a natively-buffered index actually search correctly, tolerate
 * deletions, etc.) lives separately in {@link TestNativeFlatBufferingIndexAndSearch}.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestNativeFlatBufferingGuardRails extends LuceneTestCase {

  private static final String ID_FIELD = "id";
  private static final String VECTOR_FIELD = "vector_field";

  private Random random;
  private Path indexDirPath;

  @Before
  public void beforeTest() throws Exception {
    assumeTrue("cuVS not supported", isSupported());
    random = new Random(222);
    indexDirPath = Paths.get(UUID.randomUUID().toString());
  }

  @After
  public void afterTest() throws Exception {
    if (indexDirPath == null) {
      return;
    }
    File indexDirPathFile = indexDirPath.toFile();
    if (indexDirPathFile.exists() && indexDirPathFile.isDirectory()) {
      FileUtils.deleteDirectory(indexDirPathFile);
    }
  }

  /**
   * The count-mismatch guard exists for a caller-side bookkeeping error: declaring {@code
   * numInputVectors} against the pre-filter document count instead of the number of vectors that
   * actually reach {@code addValue} (e.g. an ingest-time filter skips some documents' vector
   * field). It is unrelated to, and not triggered by, Lucene-level deletion -- see the javadoc on
   * {@link TestNativeFlatBufferingIndexAndSearch#testDeletedDocsAfterNativeFlatBufferedFlush}.
   */
  @Test
  public void testCountMismatchFromIngestTimeFilterIsRejected() throws Exception {
    int declaredNumInputVectors = 100;
    int actuallyIndexed = declaredNumInputVectors - 1; // one doc "filtered out" before addValue
    int dimension = 32;
    float[][] dataset = generateDataset(random, actuallyIndexed, dimension);

    AcceleratedHNSWParams params = new AcceleratedHNSWParams.Builder().build();
    Codec codec = new Lucene101AcceleratedHNSWCodec(params, declaredNumInputVectors);
    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(codec)
            .setUseCompoundFile(false)
            .setMaxBufferedDocs(declaredNumInputVectors + 1)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH)
            .setMergePolicy(NoMergePolicy.INSTANCE);

    try (Directory dir = FSDirectory.open(indexDirPath);
        IndexWriter writer = new IndexWriter(dir, config)) {
      for (int i = 0; i < actuallyIndexed; i++) {
        Document document = new Document();
        document.add(new StringField(ID_FIELD, Integer.toString(i), Field.Store.YES));
        document.add(new KnnFloatVectorField(VECTOR_FIELD, dataset[i], EUCLIDEAN));
        writer.addDocument(document);
      }
      IllegalStateException thrown = expectThrows(IllegalStateException.class, writer::commit);
      assertTrue(
          "unexpected message: " + thrown.getMessage(),
          thrown.getMessage().contains("numInputVectors"));
    }
  }

  /**
   * When one field's native buffer fails the count-mismatch guard, {@code flush}'s per-field loop
   * throws immediately -- a later field, even if it was populated correctly, is never reached by
   * {@code writeFieldNative} and so never gets its native buffer released there. Confirms {@code
   * NativeFlatBufferedHNSWVectorsWriter#close} still releases every field's buffer as a backstop
   * regardless of what flush reached, by asserting close does not throw even though the second
   * field's buffer was never touched during the failed flush.
   */
  @Test
  public void testCloseAfterCountMismatchReleasesEveryFieldsBuffer() throws Exception {
    int declaredNumInputVectors = 50;
    int dimension = 16;
    // fieldA is added to documents first, so it is flush()'s first (and only-attempted) field;
    // fieldB is added second and, unlike fieldA, is fully and correctly filled on every document.
    String fieldA = "vector_field_a";
    String fieldB = "vector_field_b";
    float[][] datasetA = generateDataset(random, declaredNumInputVectors - 1, dimension);
    float[][] datasetB = generateDataset(random, declaredNumInputVectors, dimension);

    AcceleratedHNSWParams params = new AcceleratedHNSWParams.Builder().build();
    Codec codec = new Lucene101AcceleratedHNSWCodec(params, declaredNumInputVectors);
    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(codec)
            .setUseCompoundFile(false)
            .setMaxBufferedDocs(declaredNumInputVectors + 1)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH)
            .setMergePolicy(NoMergePolicy.INSTANCE);

    try (Directory dir = FSDirectory.open(indexDirPath);
        IndexWriter writer = new IndexWriter(dir, config)) {
      for (int i = 0; i < declaredNumInputVectors; i++) {
        Document document = new Document();
        document.add(new StringField(ID_FIELD, Integer.toString(i), Field.Store.YES));
        // Skip fieldA on the last doc so it ends up one short of declaredNumInputVectors.
        if (i < declaredNumInputVectors - 1) {
          document.add(new KnnFloatVectorField(fieldA, datasetA[i], EUCLIDEAN));
        }
        document.add(new KnnFloatVectorField(fieldB, datasetB[i], EUCLIDEAN));
        writer.addDocument(document);
      }
      IllegalStateException thrown = expectThrows(IllegalStateException.class, writer::commit);
      assertTrue(
          "unexpected message: " + thrown.getMessage(),
          thrown.getMessage().contains("numInputVectors"));
      // The try-with-resources close() below must not throw even though fieldB's fully-populated
      // native buffer was never reached by the aborted flush() loop.
    }
  }

  /** Native flat buffering pre-sizes a single flush's buffer and cannot support sorted flushes. */
  @Test
  public void testIndexSortedSegmentIsRejected() throws Exception {
    int numDocs = 50;
    int dimension = 32;
    float[][] dataset = generateDataset(random, numDocs, dimension);

    AcceleratedHNSWParams params = new AcceleratedHNSWParams.Builder().build();
    Codec codec = new Lucene101AcceleratedHNSWCodec(params, numDocs);
    Sort indexSort = new Sort(new SortField("sort_key", SortField.Type.LONG));
    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(codec)
            .setUseCompoundFile(false)
            .setIndexSort(indexSort)
            .setMaxBufferedDocs(numDocs + 1)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH)
            .setMergePolicy(NoMergePolicy.INSTANCE);

    try (Directory dir = FSDirectory.open(indexDirPath);
        IndexWriter writer = new IndexWriter(dir, config)) {
      // KnnVectorsFormat#fieldsWriter (and so the writer's index-sort check in its constructor)
      // is invoked on the first addDocument() for the segment, not at commit() -- so the guard
      // must be expected around the whole indexing loop, not just the flush.
      IllegalArgumentException thrown =
          expectThrows(
              IllegalArgumentException.class,
              () -> {
                for (int i = 0; i < numDocs; i++) {
                  Document document = new Document();
                  document.add(new StringField(ID_FIELD, Integer.toString(i), Field.Store.YES));
                  document.add(new NumericDocValuesField("sort_key", numDocs - i));
                  document.add(new KnnFloatVectorField(VECTOR_FIELD, dataset[i], EUCLIDEAN));
                  writer.addDocument(document);
                }
              });
      assertTrue(
          "unexpected message: " + thrown.getMessage(),
          thrown.getMessage().contains("index-sorted"));
    }
  }

  /**
   * Native flat buffering supports only the unsorted single-segment flush path; merging two
   * natively-buffered segments must be rejected rather than silently mis-sizing the native buffer.
   */
  @Test
  public void testMergeIsRejected() throws Exception {
    int segmentSize = 40;
    int dimension = 32;

    AcceleratedHNSWParams params = new AcceleratedHNSWParams.Builder().build();
    Codec codec = new Lucene101AcceleratedHNSWCodec(params, segmentSize);
    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(codec)
            .setUseCompoundFile(false)
            .setMaxBufferedDocs(segmentSize + 1)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH)
            // Keep the two flushes below as separate segments; NoMergePolicy also blocks forced
            // merges, so it is swapped out before forceMerge() is called.
            .setMergePolicy(NoMergePolicy.INSTANCE)
            // Force merges to run synchronously on the calling thread, so the guard's exception
            // (or whatever IndexWriter/SegmentMerger wraps it as) surfaces directly from
            // forceMerge() instead of on a background merge thread.
            .setMergeScheduler(new SerialMergeScheduler());

    try (Directory dir = FSDirectory.open(indexDirPath);
        IndexWriter writer = new IndexWriter(dir, config)) {
      addSegment(writer, 0, segmentSize, dimension);
      writer.commit(); // segment 1: exactly segmentSize vectors, matching numInputVectors
      addSegment(writer, segmentSize, segmentSize, dimension);
      writer.commit(); // segment 2: exactly segmentSize vectors, matching numInputVectors

      writer.getConfig().setMergePolicy(new TieredMergePolicy());
      Throwable thrown = expectThrows(Throwable.class, () -> writer.forceMerge(1));
      assertTrue(
          "expected UnsupportedOperationException somewhere in the cause chain of: " + thrown,
          causedBy(thrown, UnsupportedOperationException.class));
    }
  }

  private void addSegment(IndexWriter writer, int startId, int count, int dimension)
      throws Exception {
    float[][] dataset = generateDataset(random, count, dimension);
    for (int i = 0; i < count; i++) {
      Document document = new Document();
      document.add(new StringField(ID_FIELD, Integer.toString(startId + i), Field.Store.YES));
      document.add(new KnnFloatVectorField(VECTOR_FIELD, dataset[i], EUCLIDEAN));
      writer.addDocument(document);
    }
  }

  private static boolean causedBy(Throwable t, Class<? extends Throwable> type) {
    for (Throwable cur = t; cur != null; cur = cur.getCause()) {
      if (type.isInstance(cur)) {
        return true;
      }
      for (Throwable suppressed : cur.getSuppressed()) {
        if (causedBy(suppressed, type)) {
          return true;
        }
      }
    }
    return false;
  }
}

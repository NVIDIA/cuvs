/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

public class TestCagraHnswBulkIndexWriterValidation extends LuceneTestCase {

  @Test
  public void testSliceCountValidationIsBoundedAndExact() {
    for (int total : new int[] {-1, 0}) {
      expectThrows(
          IllegalArgumentException.class, () -> CagraHnswBulkIndexWriter.sliceEvenly(total, 1));
    }
    expectThrows(IllegalArgumentException.class, () -> CagraHnswBulkIndexWriter.sliceEvenly(1, 0));
    expectThrows(IllegalArgumentException.class, () -> CagraHnswBulkIndexWriter.sliceEvenly(1, 2));
    expectThrows(
        IllegalArgumentException.class,
        () -> CagraHnswBulkIndexWriter.sliceEvenly(1, Integer.MAX_VALUE));
    expectThrows(
        IllegalArgumentException.class,
        () -> CagraHnswBulkIndexWriter.sliceEvenly(IndexWriter.MAX_DOCS + 1, 2));

    List<int[]> slices = CagraHnswBulkIndexWriter.sliceEvenly(5, 3);
    assertEquals(3, slices.size());
    assertArrayEquals(new int[] {0, 2}, slices.get(0));
    assertArrayEquals(new int[] {2, 2}, slices.get(1));
    assertArrayEquals(new int[] {4, 1}, slices.get(2));
  }

  @Test
  public void testPublicBuildRejectsEmptySourceWithoutCreatingTarget() throws Exception {
    Path target = createTempDir().resolve("empty-index");
    IllegalArgumentException failure =
        expectThrows(
            IllegalArgumentException.class,
            () -> CagraHnswBulkIndexWriter.build(new FixedShapeSource(0), config(target, 1)));
    assertTrue(failure.getMessage(), failure.getMessage().contains("must be > 0"));
    assertFalse(Files.exists(target));
  }

  @Test
  public void testPublicBuildRejectsExtremeSegmentsWithoutCreatingTarget() throws Exception {
    Path target = createTempDir().resolve("too-many-segments-index");
    IllegalArgumentException failure =
        expectThrows(
            IllegalArgumentException.class,
            () ->
                CagraHnswBulkIndexWriter.build(
                    new FixedShapeSource(1), config(target, Integer.MAX_VALUE)));
    assertTrue(failure.getMessage(), failure.getMessage().contains("exceeds bulk vector count"));
    assertFalse(Files.exists(target));
  }

  @Test
  public void testPublicBuildRejectsTotalAboveLuceneLimitWithoutCreatingTarget() throws Exception {
    Path target = createTempDir().resolve("too-many-documents-index");
    IllegalArgumentException failure =
        expectThrows(
            IllegalArgumentException.class,
            () ->
                CagraHnswBulkIndexWriter.build(
                    new FixedShapeSource(IndexWriter.MAX_DOCS + 1), config(target, 2)));
    assertTrue(failure.getMessage(), failure.getMessage().contains("index-wide document limit"));
    assertFalse(Files.exists(target));
  }

  @Test
  public void testOneShotBuildRequiresTarget() {
    IllegalArgumentException failure =
        expectThrows(
            IllegalArgumentException.class,
            () -> CagraHnswBulkIndexWriter.build(new FixedShapeSource(1), config()));
    assertTrue(failure.getMessage(), failure.getMessage().contains("targetDirectory"));
  }

  @Test
  public void testTargetIsStoredAbsoluteAndNormalized() {
    Path supplied = Path.of("build-area", "..", "index");
    assertEquals(supplied.toAbsolutePath().normalize(), config(supplied, 1).targetDirectory());
  }

  @Test
  public void testFbinCannotAlsoBeTheTargetDirectory() throws Exception {
    Path source = createTempDir().resolve("source.fbin");
    Files.write(source, new byte[8]);
    IllegalArgumentException failure =
        expectThrows(
            IllegalArgumentException.class,
            () -> CagraHnswBulkIndexWriter.indexFbin(source, config(source, 1)));
    assertTrue(failure.getMessage(), failure.getMessage().contains("must not be the source FBIN"));
  }

  @Test
  public void testExactCountAcceptsLuceneSegmentLimitWithoutOverflow() throws Exception {
    try (Directory directory = newDirectory()) {
      CagraHnswBulkIndexWriter writer =
          new CagraHnswBulkIndexWriter(
              directory, new IndexWriterConfig(), config(), IndexWriter.MAX_DOCS);
      writer.abort();
    }
  }

  @Test
  public void testExactCountRejectsValuesAboveLuceneIndexLimitBeforeWriterCreation()
      throws Exception {
    for (int count : new int[] {IndexWriter.MAX_DOCS + 1, Integer.MAX_VALUE}) {
      try (Directory directory = newDirectory()) {
        IllegalArgumentException failure =
            expectThrows(
                IllegalArgumentException.class,
                () ->
                    new CagraHnswBulkIndexWriter(
                        directory, new IndexWriterConfig(), config(), count));
        assertTrue(
            failure.getMessage(), failure.getMessage().contains("index-wide document limit"));
        try (var lock = directory.obtainLock(IndexWriter.WRITE_LOCK_NAME)) {
          assertNotNull(lock);
        }
      }
    }
  }

  @Test
  public void testFailedExplicitCommitIsNotRetriedByClose() throws Exception {
    try (FailFirstCommitRenameDirectory directory =
        new FailFirstCommitRenameDirectory(newDirectory())) {
      IndexWriterConfig owned =
          CagraHnswBulkIndexWriter.ownedIndexWriterConfig(
              new IndexWriterConfig(), Codec.getDefault(), 1);
      IOException failure =
          expectThrows(
              IOException.class,
              () -> {
                try (IndexWriter writer = new IndexWriter(directory, owned)) {
                  Document document = new Document();
                  document.add(new StringField("id", "0", Field.Store.NO));
                  writer.addDocument(document);
                  writer.commit();
                }
              });

      assertEquals("first publication rename failed", failure.getMessage());
      assertEquals(1, directory.publicationRenameAttempts);
      assertFalse(DirectoryReader.indexExists(directory));
    }
  }

  private static CagraHnswBulkIndexWriter.Config config() {
    return CagraHnswBulkIndexWriter.Config.builder()
        .field("vector", 2, EUCLIDEAN)
        .graphBuild(new AcceleratedHNSWParams.Builder().build())
        .build();
  }

  private static CagraHnswBulkIndexWriter.Config config(Path target, int segments) {
    return CagraHnswBulkIndexWriter.Config.builder()
        .field("vector", 2, EUCLIDEAN)
        .graphBuild(new AcceleratedHNSWParams.Builder().build())
        .segments(segments, false)
        .targetDirectory(target)
        .build();
  }

  private record FixedShapeSource(int size) implements VectorSource {
    @Override
    public int dimensions() {
      return 2;
    }

    @Override
    public void get(int index, float[] dst) {
      throw new AssertionError("source must not be consumed during validation");
    }

    @Override
    public void close() {}
  }

  private static final class FailFirstCommitRenameDirectory extends FilterDirectory {
    private int publicationRenameAttempts;

    private FailFirstCommitRenameDirectory(Directory in) {
      super(in);
    }

    @Override
    public void rename(String source, String dest) throws IOException {
      if (source.startsWith("pending_segments") && dest.startsWith("segments")) {
        publicationRenameAttempts++;
        if (publicationRenameAttempts == 1) {
          throw new IOException("first publication rename failed");
        }
      }
      super.rename(source, dest);
    }
  }
}

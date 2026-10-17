/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.closeCuVSResourcesInstance;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.setCuVSResourcesInstance;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.spi.CuVSProvider;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.HashSet;
import java.util.List;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.Test;

@SuppressSysoutChecks(bugUrl = "")
public class TestCuVSQueryMatrixLifecycle extends LuceneTestCase {

  private static final String VECTOR_FIELD = "vector";
  private static final int DOCUMENT_COUNT = 1024;
  private static final int DIMENSIONS = 128;
  private static final int QUERY_DOCUMENT = 137;
  private static final int TOP_K = 10;
  private static final int QUERY_REPETITIONS = 128;

  @Test
  public void testRepeatedCagraQueriesReleaseWorkspaceAllocations() throws Throwable {
    assumeTrue("cuVS not supported", isSupported());

    float[][] vectors = deterministicVectors();
    Path memoryCsv = createTempDir("cagra-query-memory").resolve("allocations.csv");

    try (Directory directory = new ByteBuffersDirectory()) {
      buildSingleSegmentIndex(directory, vectors);
      runRepeatedQueriesWithMemoryTracking(directory, vectors[QUERY_DOCUMENT], memoryCsv);
    }

    MemoryTotals workspace = finalMemoryTotals(memoryCsv, "workspace");
    assertTrue("the tracker did not observe workspace allocations", workspace.allocated() > 0);
    assertEquals("CAGRA queries retained workspace memory", 0, workspace.current());
    assertEquals(
        "workspace allocations and deallocations are unbalanced",
        workspace.allocated(),
        workspace.freed());
  }

  private static void buildSingleSegmentIndex(Directory directory, float[][] vectors)
      throws Exception {
    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(new CuVS2510GPUSearchCodec())
            .setMergePolicy(NoMergePolicy.INSTANCE)
            .setMaxBufferedDocs(vectors.length + 1);
    try (IndexWriter writer = new IndexWriter(directory, config)) {
      for (float[] vector : vectors) {
        Document document = new Document();
        document.add(new KnnFloatVectorField(VECTOR_FIELD, vector, EUCLIDEAN));
        writer.addDocument(document);
      }
      writer.commit();
    }
  }

  private static void runRepeatedQueriesWithMemoryTracking(
      Directory directory, float[] queryVector, Path memoryCsv) throws Throwable {
    try (CuVSResources trackedResources =
        CuVSResources.create(CuVSProvider.tempDirectory(), memoryCsv, Duration.ofMillis(1))) {
      setCuVSResourcesInstance(new NonClosingResources(trackedResources));
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        LeafReader leaf = getOnlyLeafReader(reader);
        assertCagraIndexLoaded(leaf);
        for (int i = 0; i < QUERY_REPETITIONS; i++) {
          TopKnnCollector collector = new TopKnnCollector(TOP_K, Integer.MAX_VALUE);
          leaf.searchNearestVectors(VECTOR_FIELD, queryVector, collector, null);
          assertCorrectResults(collector.topDocs().scoreDocs);
        }
      } finally {
        // DirectoryReader closes the provider's wrapper. Remove any value left by an early failure
        // without closing the underlying tracker a second time.
        setCuVSResourcesInstance(null);
        closeCuVSResourcesInstance();
      }
    }
  }

  private static void assertCagraIndexLoaded(LeafReader leaf) {
    KnnVectorsReader vectorReader = ((CodecReader) leaf).getVectorReader();
    if (vectorReader instanceof PerFieldKnnVectorsFormat.FieldsReader fieldsReader) {
      vectorReader = fieldsReader.getFieldReader(VECTOR_FIELD);
    }
    assertTrue(
        "expected the cuVS GPU vectors reader", vectorReader instanceof CuVS2510GPUVectorsReader);
    CuVS2510GPUVectorsReader cuvsReader = (CuVS2510GPUVectorsReader) vectorReader;
    assertNotNull("expected a CAGRA index", cuvsReader.getCagraIndexForField(VECTOR_FIELD));
  }

  private static void assertCorrectResults(ScoreDoc[] hits) {
    assertEquals("unexpected result count", TOP_K, hits.length);
    assertEquals("the queried document must rank first", QUERY_DOCUMENT, hits[0].doc);
    HashSet<Integer> uniqueDocIds = new HashSet<>();
    for (ScoreDoc hit : hits) {
      assertTrue("duplicate result " + hit.doc, uniqueDocIds.add(hit.doc));
    }
  }

  private static float[][] deterministicVectors() {
    float[][] vectors = new float[DOCUMENT_COUNT][DIMENSIONS];
    for (int row = 0; row < DOCUMENT_COUNT; row++) {
      for (int column = 0; column < DIMENSIONS; column++) {
        vectors[row][column] = ((row + 1L) * (column + 3L) % 1031L) / 1031.0f;
      }
    }
    return vectors;
  }

  private static MemoryTotals finalMemoryTotals(Path csv, String resourceName) throws Exception {
    List<String> lines = Files.readAllLines(csv);
    assertTrue("memory tracker did not write a sample", lines.size() >= 2);
    String[] header = lines.getFirst().split(",", -1);
    String[] finalSample = lines.getLast().split(",", -1);
    return new MemoryTotals(
        csvLong(header, finalSample, resourceName + "_current"),
        csvLong(header, finalSample, resourceName + "_total_alloc"),
        csvLong(header, finalSample, resourceName + "_total_freed"));
  }

  private static long csvLong(String[] header, String[] sample, String columnName) {
    for (int i = 0; i < header.length; i++) {
      if (columnName.equals(header[i])) {
        return Long.parseLong(sample[i]);
      }
    }
    fail("memory tracker CSV is missing column " + columnName);
    throw new AssertionError("unreachable");
  }

  private record MemoryTotals(long current, long allocated, long freed) {}

  /** Prevents a Lucene reader from closing the tracker before its final CSV sample is written. */
  private static final class NonClosingResources implements CuVSResources {
    private final CuVSResources delegate;

    private NonClosingResources(CuVSResources delegate) {
      this.delegate = delegate;
    }

    @Override
    public ScopedAccess access() {
      return delegate.access();
    }

    @Override
    public int deviceId() {
      return delegate.deviceId();
    }

    @Override
    public void close() {}

    @Override
    public void setWorkspacePool(long initialSizeBytes) {
      delegate.setWorkspacePool(initialSizeBytes);
    }

    @Override
    public Path tempDirectory() {
      return delegate.tempDirectory();
    }
  }
}

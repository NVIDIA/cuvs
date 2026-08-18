/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.generateDataset;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
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
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/**
 * Positive-path functional coverage for native flat buffering ({@code
 * AcceleratedHNSWParams.numInputVectors}) beyond {@code TestNativeFlatVectorsWriterRoundTrip},
 * which only checks that the flat {@code .vec} file round-trips -- not that the resulting index is
 * actually searchable, tolerates deletions, or composes correctly with the odd-graph-degree fix.
 *
 * <p>The negative/guard-rail paths (count mismatch, index-sorted segments, merges) are covered
 * separately in {@link TestNativeFlatBufferingGuardRails}.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestNativeFlatBufferingIndexAndSearch extends LuceneTestCase {

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

  /** A natively-buffered index must still be searchable through the normal Lucene KNN query API. */
  @Test
  public void testIndexAndSearch() throws Exception {
    int numDocs = 500;
    int dimension = 32;
    int topK = 10;
    float[][] dataset = generateDataset(random, numDocs, dimension);

    buildNativeFlatBufferedIndex(numDocs, dimension, dataset);

    try (Directory dir = FSDirectory.open(indexDirPath);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      assertEquals("expected a single native-flat-buffered segment", 1, reader.leaves().size());

      IndexSearcher searcher = new IndexSearcher(reader);
      float[] queryVector = generateDataset(random, 1, dimension)[0];
      TopDocs results =
          searcher.search(new KnnFloatVectorQuery(VECTOR_FIELD, queryVector, topK), topK);

      assertEquals("expected topK results", topK, results.scoreDocs.length);
      for (var scoreDoc : results.scoreDocs) {
        String id = searcher.storedFields().document(scoreDoc.doc).get(ID_FIELD);
        int idValue = Integer.parseInt(id);
        assertTrue("returned id out of range: " + id, idValue >= 0 && idValue < numDocs);
      }
    }
  }

  /**
   * Deletion is orthogonal to native flat buffering: {@code IndexWriter.deleteDocuments} only
   * updates Lucene's liveDocs bitset at search time -- it never touches {@code FieldWriter} or the
   * native host matrix, so it cannot trip (and isn't meant to be caught by) the count-mismatch
   * guard rail tested in {@link TestNativeFlatBufferingGuardRails}. This test instead confirms that
   * deletions applied after a natively-buffered flush are still honored correctly at search time.
   */
  @Test
  public void testDeletedDocsAfterNativeFlatBufferedFlush() throws Exception {
    int numDocs = 300;
    int dimension = 32;
    float[][] dataset = generateDataset(random, numDocs, dimension);

    AcceleratedHNSWParams params =
        new AcceleratedHNSWParams.Builder().withNumInputVectors(numDocs).build();
    Codec codec = new Lucene101AcceleratedHNSWCodec(params);
    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(codec)
            .setUseCompoundFile(false)
            .setMaxBufferedDocs(numDocs + 1)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH)
            .setMergePolicy(NoMergePolicy.INSTANCE);

    try (Directory dir = FSDirectory.open(indexDirPath);
        IndexWriter writer = new IndexWriter(dir, config)) {
      for (int i = 0; i < numDocs; i++) {
        Document document = new Document();
        document.add(new StringField(ID_FIELD, Integer.toString(i), Field.Store.YES));
        document.add(new KnnFloatVectorField(VECTOR_FIELD, dataset[i], EUCLIDEAN));
        writer.addDocument(document);
      }
      writer.commit(); // single natively-buffered flush: FieldWriter's count matches numDocs

      // Delete every 3rd doc. No new vectors are added, so this does not trigger another flush of
      // the vector field and cannot interact with the numInputVectors hint.
      for (int i = 0; i < numDocs; i += 3) {
        writer.deleteDocuments(new Term(ID_FIELD, Integer.toString(i)));
      }
      writer.commit();
    }

    try (Directory dir = FSDirectory.open(indexDirPath);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      assertEquals(
          "expected the deletions to land in the same single segment", 1, reader.leaves().size());
      assertTrue("expected some deleted docs", reader.numDeletedDocs() > 0);

      IndexSearcher searcher = new IndexSearcher(reader);
      TopDocs results =
          searcher.search(new KnnFloatVectorQuery(VECTOR_FIELD, dataset[0], numDocs), numDocs);
      for (var scoreDoc : results.scoreDocs) {
        String id = searcher.storedFields().document(scoreDoc.doc).get(ID_FIELD);
        assertNotEquals(
            "deleted doc id=" + id + " was still returned by search", 0, Integer.parseInt(id) % 3);
      }
    }
  }

  /**
   * The M = ceil(cagraGraphDegree / 2) fix ({@link TestAcceleratedHNSWOddGraphDegree}) must also
   * hold on the native-flat-buffered write path ({@code writeFieldNative}), which is a distinct
   * call path from the heap-buffered one that test exercises.
   */
  @Test
  public void testOddGraphDegreeWithNativeFlatBuffering() throws Exception {
    int numDocs = 200;
    int dimension = 32;
    int oddGraphDegree = 63;
    float[][] dataset = generateDataset(random, numDocs, dimension);

    AcceleratedHNSWParams params =
        new AcceleratedHNSWParams.Builder()
            .withStrategy(AcceleratedHNSWParams.Strategy.CUSTOM)
            .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
            .withIntermediateGraphDegree(128)
            .withGraphDegree(oddGraphDegree)
            .withNumInputVectors(numDocs)
            .build();

    buildNativeFlatBufferedIndex(numDocs, dimension, dataset, params);

    try (Directory dir = FSDirectory.open(indexDirPath);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      TopDocs results = searcher.search(new KnnFloatVectorQuery(VECTOR_FIELD, dataset[0], 5), 5);
      assertEquals(5, results.scoreDocs.length);
    }
  }

  private void buildNativeFlatBufferedIndex(int numDocs, int dimension, float[][] dataset)
      throws Exception {
    buildNativeFlatBufferedIndex(
        numDocs,
        dimension,
        dataset,
        new AcceleratedHNSWParams.Builder().withNumInputVectors(numDocs).build());
  }

  private void buildNativeFlatBufferedIndex(
      int numDocs, int dimension, float[][] dataset, AcceleratedHNSWParams params)
      throws Exception {
    Codec codec = new Lucene101AcceleratedHNSWCodec(params);
    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(codec)
            .setUseCompoundFile(false)
            .setMaxBufferedDocs(numDocs + 1)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH)
            .setMergePolicy(NoMergePolicy.INSTANCE);

    try (Directory dir = FSDirectory.open(indexDirPath);
        IndexWriter writer = new IndexWriter(dir, config)) {
      for (int i = 0; i < numDocs; i++) {
        Document document = new Document();
        document.add(new StringField(ID_FIELD, Integer.toString(i), Field.Store.YES));
        document.add(new KnnFloatVectorField(VECTOR_FIELD, dataset[i], EUCLIDEAN));
        writer.addDocument(document);
      }
      writer.commit();
    }
  }
}

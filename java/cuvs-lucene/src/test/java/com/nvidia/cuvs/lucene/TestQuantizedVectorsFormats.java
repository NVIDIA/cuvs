/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.assertVectorsKeepTheirDocuments;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.COSINE;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import com.carrotsearch.randomizedtesting.annotations.Name;
import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.index.BaseKnnVectorsFormatTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.tests.util.TestUtil;
import org.junit.BeforeClass;
import org.junit.Ignore;

@SuppressSysoutChecks(bugUrl = "")
public class TestQuantizedVectorsFormats extends BaseKnnVectorsFormatTestCase {

  private static final Logger log = Logger.getLogger(TestQuantizedVectorsFormats.class.getName());

  private static KnnVectorsFormat knnVectorsFormat;

  public TestQuantizedVectorsFormats(@Name("knnVectorsWriter") KnnVectorsFormat knnVectorsFormat) {
    TestQuantizedVectorsFormats.knnVectorsFormat = knnVectorsFormat;
  }

  @ParametersFactory
  public static List<Object[]> parameters() {
    return Arrays.asList(
        new Object[][] {
          {new LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat()},
          {new LuceneAcceleratedHNSWScalarQuantizedVectorsFormat()}
        });
  }

  @BeforeClass
  public static void beforeClass() {
    assumeTrue("cuVS is not supported so skipping these tests", isSupported());
  }

  @Override
  protected Codec getCodec() {
    log.log(Level.FINE, "Running tests for: " + knnVectorsFormat.getName());
    return TestUtil.alwaysKnnVectorsFormat(knnVectorsFormat);
  }

  public void testMergeTwoSegsWithASingleDocPerSeg() throws Exception {
    final int R = 2, D = 128;
    float[][] f = new float[R][D];
    final String F = "f";
    for (int i = 0; i < R; i++) {
      f[i] = randomVector(D);
    }

    try (Directory dir = newDirectory(new ByteBuffersDirectory());
        IndexWriter w = new IndexWriter(dir, newIndexWriterConfig())) {
      for (int i = 0; i < R; i++) {
        Document doc = new Document();
        doc.add(new StringField("id", String.valueOf(i), Field.Store.YES));
        doc.add(new KnnFloatVectorField(F, f[i], EUCLIDEAN));
        w.addDocument(doc);
        w.commit();
      }
      w.flush();

      try (DirectoryReader reader = DirectoryReader.open(w)) {
        List<LeafReaderContext> subReaders = reader.leaves();
        assertEquals(2, subReaders.size());
        for (int i = 0; i < R; i++) {
          assertEquals(1, subReaders.get(i).reader().getFloatVectorValues(F).size());
        }
      }
      w.forceMerge(1);

      try (DirectoryReader reader = DirectoryReader.open(w)) {
        LeafReader r = getOnlyLeafReader(reader);
        assertVectorsKeepTheirDocuments(r, F, f);
      }
    }
  }

  public void testTwoVectorFieldsPerDoc() throws Exception {
    final int R = 2, D = 128;
    final String F1 = "f1", F2 = "f2";
    float[][] f1 = new float[R][D];
    float[][] f2 = new float[R][D];

    for (int i = 0; i < R; i++) {
      f1[i] = randomVector(D);
      f2[i] = randomVector(D);
    }

    try (Directory dir = newDirectory(new ByteBuffersDirectory());
        IndexWriter w = new IndexWriter(dir, newIndexWriterConfig())) {

      for (int i = 0; i < R; i++) {
        Document doc = new Document();
        doc.add(new StringField("id", String.valueOf(i), Field.Store.YES));
        doc.add(new KnnFloatVectorField(F1, f1[i], EUCLIDEAN));
        doc.add(new KnnFloatVectorField(F2, f2[i], EUCLIDEAN));
        w.addDocument(doc);
      }
      w.forceMerge(1);

      try (DirectoryReader reader = DirectoryReader.open(w)) {
        LeafReader r = getOnlyLeafReader(reader);
        assertVectorsKeepTheirDocuments(r, F1, f1);
        assertVectorsKeepTheirDocuments(r, F2, f2);
      }
    }
  }

  public void testCosineSimilarity() throws Exception {
    final int R = 2, D = 128;
    final String F = "f";
    float[][] f = new float[R][D];
    for (int i = 0; i < R; i++) {
      f[i] = randomVector(D);
    }

    try (Directory dir = newDirectory(new ByteBuffersDirectory());
        IndexWriter w = new IndexWriter(dir, newIndexWriterConfig())) {

      for (int i = 0; i < R; i++) {
        Document doc = new Document();
        doc.add(new StringField("id", String.valueOf(i), Field.Store.NO));
        doc.add(new KnnFloatVectorField(F, f[i], COSINE));
        w.addDocument(doc);
      }
      w.forceMerge(1);

      try (DirectoryReader reader = DirectoryReader.open(w)) {
        LeafReader r = getOnlyLeafReader(reader);

        FloatVectorValues values = r.getFloatVectorValues(F);
        assertNotNull(values);
        assertEquals(R, values.size());

        float[] queryVector = randomVector(D);
        var topDocs = r.searchNearestVectors(F, queryVector, 2, null, 10);
        assertTrue("Should return at least one result", topDocs.scoreDocs.length > 0);
        assertTrue("Scores should be non-negative", topDocs.scoreDocs[0].score >= 0);
      }
    }
  }

  public void testForceMergeUsesOnlyLiveSparseVectors() throws Exception {
    final String vectorField = "vector";
    final int dimensions = 129;
    Map<String, float[]> expected = new LinkedHashMap<>();

    try (Directory directory = newDirectory(new ByteBuffersDirectory())) {
      try (IndexWriter writer = new IndexWriter(directory, newIndexWriterConfig())) {
        for (int segment = 0; segment < 3; segment++) {
          for (int row = 0; row < 5; row++) {
            String id = segment + "-" + row;
            Document document = new Document();
            document.add(new StringField("id", id, Field.Store.YES));
            if (row < 4) {
              float[] vector = deterministicVector(segment * 5 + row, dimensions);
              document.add(new KnnFloatVectorField(vectorField, vector, EUCLIDEAN));
              if (row != 1) {
                expected.put(id, vector);
              }
            }
            writer.addDocument(document);
          }
          writer.commit();
        }
        for (int segment = 0; segment < 3; segment++) {
          writer.deleteDocuments(new Term("id", segment + "-1"));
        }
        writer.commit();
        writer.forceMerge(1);
      }

      TestUtil.checkIndex(directory);
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        LeafReader leaf = getOnlyLeafReader(reader);
        FloatVectorValues values = leaf.getFloatVectorValues(vectorField);
        assertNotNull(values);
        assertEquals(expected.size(), values.size());

        Set<String> seen = new HashSet<>();
        for (int ordinal = 0; ordinal < values.size(); ordinal++) {
          int documentId = values.ordToDoc(ordinal);
          String id = leaf.storedFields().document(documentId).get("id");
          assertTrue("Unexpected or duplicate vector for " + id, seen.add(id));
          assertArrayEquals(expected.get(id), values.vectorValue(ordinal), 0.0f);
        }
        assertEquals(expected.keySet(), seen);

        for (Map.Entry<String, float[]> entry : expected.entrySet()) {
          var hits =
              leaf.searchNearestVectors(
                  vectorField, entry.getValue(), expected.size(), null, 1_000);
          boolean found = false;
          for (var hit : hits.scoreDocs) {
            found |= entry.getKey().equals(leaf.storedFields().document(hit.doc).get("id"));
          }
          assertTrue("Exact vector was not searchable for " + entry.getKey(), found);
        }
      }
    }
  }

  public void testForceMergeWithZeroLiveVectors() throws Exception {
    assertTrivialLiveVectorMerge(0);
  }

  public void testForceMergeWithOneLiveVector() throws Exception {
    assertTrivialLiveVectorMerge(1);
  }

  private void assertTrivialLiveVectorMerge(int liveVectors) throws Exception {
    final String vectorField = "vector";
    final int dimensions = 129;
    try (Directory directory = newDirectory(new ByteBuffersDirectory())) {
      try (IndexWriter writer = new IndexWriter(directory, newIndexWriterConfig())) {
        for (int id = 0; id < 3; id++) {
          Document vectorDocument = new Document();
          vectorDocument.add(new StringField("id", "vector-" + id, Field.Store.YES));
          vectorDocument.add(
              new KnnFloatVectorField(vectorField, deterministicVector(id, dimensions), EUCLIDEAN));
          writer.addDocument(vectorDocument);

          Document sparseDocument = new Document();
          sparseDocument.add(new StringField("id", "sparse-" + id, Field.Store.YES));
          writer.addDocument(sparseDocument);
          writer.commit();
        }
        for (int id = liveVectors; id < 3; id++) {
          writer.deleteDocuments(new Term("id", "vector-" + id));
        }
        writer.commit();
        writer.forceMerge(1);
      }

      TestUtil.checkIndex(directory);
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        LeafReader leaf = getOnlyLeafReader(reader);
        FloatVectorValues values = leaf.getFloatVectorValues(vectorField);
        if (liveVectors == 0) {
          assertTrue(values == null || values.size() == 0);
        } else {
          assertNotNull(values);
          assertEquals(1, values.size());
          assertEquals("vector-0", leaf.storedFields().document(values.ordToDoc(0)).get("id"));
          assertArrayEquals(deterministicVector(0, dimensions), values.vectorValue(0), 0.0f);
        }
      }
    }
  }

  private static float[] deterministicVector(int id, int dimensions) {
    float[] vector = new float[dimensions];
    for (int dimension = 0; dimension < dimensions; dimension++) {
      vector[dimension] = id * 10.0f + dimension * 0.01f;
    }
    return vector;
  }

  @Override
  protected VectorEncoding randomVectorEncoding() {
    return VectorEncoding.FLOAT32;
  }

  @Ignore
  @Override
  public void testByteVectorScorerIteration() {}

  @Ignore
  @Override
  public void testEmptyByteVectorData() {}

  @Ignore
  @Override
  public void testMergingWithDifferentByteKnnFields() {}

  @Ignore
  @Override
  public void testMismatchedFields() {}

  @Ignore
  @Override
  public void testRandomBytes() {}

  @Ignore
  @Override
  public void testSortedIndexBytes() {}
}

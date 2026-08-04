/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.COSINE;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import com.carrotsearch.randomizedtesting.annotations.Name;
import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import java.util.Arrays;
import java.util.List;
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
        doc.add(new StringField("id", String.valueOf(i), Field.Store.NO));
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
        FloatVectorValues values = r.getFloatVectorValues(F);
        assertNotNull(values);
        assertEquals(R, values.size());
        for (int i = 0; i < R; i++) {
          assertArrayEquals(f[i], values.vectorValue(i), 0.0f);
        }
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
        doc.add(new StringField("id", String.valueOf(i), Field.Store.NO));
        doc.add(new KnnFloatVectorField(F1, f1[i], EUCLIDEAN));
        doc.add(new KnnFloatVectorField(F2, f2[i], EUCLIDEAN));
        w.addDocument(doc);
      }
      w.forceMerge(1);

      try (DirectoryReader reader = DirectoryReader.open(w)) {
        LeafReader r = getOnlyLeafReader(reader);
        FloatVectorValues values = r.getFloatVectorValues(F1);
        assertNotNull(values);
        assertEquals(R, values.size());
        for (int i = 0; i < R; i++) {
          assertArrayEquals(f1[i], values.vectorValue(i), 0.0f);
        }

        values = r.getFloatVectorValues(F2);
        assertNotNull(values);
        assertEquals(R, values.size());
        for (int i = 0; i < R; i++) {
          assertArrayEquals(f2[i], values.vectorValue(i), 0.0f);
        }
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

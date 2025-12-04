/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import java.util.List;
import org.apache.lucene.codecs.Codec;
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
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.index.BaseKnnVectorsFormatTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.tests.util.TestUtil;
import org.junit.BeforeClass;
import org.junit.Ignore;

@SuppressSysoutChecks(bugUrl = "")
public class TestLucene99AcceleratedHNSWVectorsFormat extends BaseKnnVectorsFormatTestCase {

  @BeforeClass
  public static void beforeClass() {
    assumeTrue("cuVS is not supported", Lucene99AcceleratedHNSWVectorsFormat.supported());
  }

  @Override
  protected Codec getCodec() {
    return TestUtil.alwaysKnnVectorsFormat(new Lucene99AcceleratedHNSWVectorsFormat());
  }

  public void testMergeTwoSegsWithASingleDocPerSeg() throws Exception {
    float[][] f = new float[][] {randomVector(384), randomVector(384)};
    try (Directory dir = newDirectory();
        IndexWriter w = new IndexWriter(dir, newIndexWriterConfig())) {
      Document doc1 = new Document();
      doc1.add(new StringField("id", "0", Field.Store.NO));
      doc1.add(new KnnFloatVectorField("f", f[0], EUCLIDEAN));
      w.addDocument(doc1);
      w.commit();
      Document doc2 = new Document();
      doc2.add(new StringField("id", "1", Field.Store.NO));
      doc2.add(new KnnFloatVectorField("f", f[1], EUCLIDEAN));
      w.addDocument(doc2);
      w.flush();
      w.commit();

      // sanity - verify one doc per leaf
      try (DirectoryReader reader = DirectoryReader.open(w)) {
        List<LeafReaderContext> subReaders = reader.leaves();
        assertEquals(2, subReaders.size());
        assertEquals(1, subReaders.get(0).reader().getFloatVectorValues("f").size());
        assertEquals(1, subReaders.get(1).reader().getFloatVectorValues("f").size());
      }

      // now merge to a single segment
      w.forceMerge(1);

      // verify merged content
      try (DirectoryReader reader = DirectoryReader.open(w)) {
        LeafReader r = getOnlyLeafReader(reader);
        FloatVectorValues values = r.getFloatVectorValues("f");
        assertNotNull(values);
        assertEquals(2, values.size());
        assertArrayEquals(f[0], values.vectorValue(0), 0.0f);
        assertArrayEquals(f[1], values.vectorValue(1), 0.0f);
      }
    }
  }

  // Basic test for multiple vectors fields per document
  public void testTwoVectorFieldsPerDoc() throws Exception {
    float[][] f1 = new float[][] {randomVector(384), randomVector(384)};
    float[][] f2 = new float[][] {randomVector(384), randomVector(384)};
    try (Directory dir = newDirectory();
        IndexWriter w = new IndexWriter(dir, newIndexWriterConfig())) {
      Document doc1 = new Document();
      doc1.add(new StringField("id", "0", Field.Store.NO));
      doc1.add(new KnnFloatVectorField("f1", f1[0], EUCLIDEAN));
      doc1.add(new KnnFloatVectorField("f2", f2[0], EUCLIDEAN));
      w.addDocument(doc1);
      Document doc2 = new Document();
      doc2.add(new StringField("id", "1", Field.Store.NO));
      doc2.add(new KnnFloatVectorField("f1", f1[1], EUCLIDEAN));
      doc2.add(new KnnFloatVectorField("f2", f2[1], EUCLIDEAN));
      w.addDocument(doc2);
      w.forceMerge(1);

      try (DirectoryReader reader = DirectoryReader.open(w)) {
        LeafReader r = getOnlyLeafReader(reader);
        FloatVectorValues values = r.getFloatVectorValues("f1");
        assertNotNull(values);
        assertEquals(2, values.size());
        assertArrayEquals(f1[0], values.vectorValue(0), 0.0f);
        assertArrayEquals(f1[1], values.vectorValue(1), 0.0f);

        values = r.getFloatVectorValues("f2");
        assertNotNull(values);
        assertEquals(2, values.size());
        assertArrayEquals(f2[0], values.vectorValue(0), 0.0f);
        assertArrayEquals(f2[1], values.vectorValue(1), 0.0f);

        // opportunistically check boundary condition - search with a 0 topK
        var topDocs = r.searchNearestVectors("f1", randomVector(384), 0, null, 10);
        assertEquals(0, topDocs.scoreDocs.length);
        assertEquals(0, topDocs.totalHits.value());
      }
    }
  }

  @Override
  // Overriding this method from superclass for the tests to only use float vector encoding
  protected VectorEncoding randomVectorEncoding() {
    return VectorEncoding.FLOAT32;
  }

  @Ignore
  @Override
  // Ignoring this test from superclass as we do not support byte vectors
  public void testByteVectorScorerIteration() {}

  @Ignore
  @Override
  // Ignoring this test from superclass as we do not support byte vectors
  public void testEmptyByteVectorData() {}

  @Ignore
  @Override
  // Ignoring this test from superclass as we do not support byte vectors
  public void testMergingWithDifferentByteKnnFields() {}

  @Ignore
  @Override
  // Ignoring this test from superclass as we do not support byte vectors
  public void testMismatchedFields() {}

  @Ignore
  @Override
  // Ignoring this test from superclass as we do not support byte vectors
  public void testRandomBytes() {}

  @Ignore
  @Override
  // Ignoring this test from superclass as we do not support byte vectors
  public void testSortedIndexBytes() {}
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.assertVectorsKeepTheirDocuments;
import static org.apache.lucene.index.VectorSimilarityFunction.COSINE;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import com.carrotsearch.randomizedtesting.annotations.Name;
import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.tests.util.TestUtil;

@SuppressSysoutChecks(bugUrl = "")
public class TestQuantizedVectorsFormats extends BaseCuVSKnnVectorsFormatTestCase {

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
          {
            CuVSCodecs.acceleratedHNSWScalarQuantizedFormat(
                new AcceleratedHNSWParams.Builder().build())
          }
        });
  }

  /**
   * Only the scalar-quantized format of Lucene 10.4+ keeps a quantized copy it can rebuild float
   * vectors from; the binary-quantized format stores full-precision vectors only. Matched by name,
   * because the class does not exist on earlier releases.
   */
  @Override
  protected boolean supportsFloatVectorFallback() {
    return "Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat"
        .equals(knnVectorsFormat.getName());
  }

  // The format quantizes to 7 bits (ScalarEncoding.SEVEN_BIT). Since Lucene 10.4; not marked
  // @Override, so that it compiles against earlier releases.
  protected int getQuantizationBits() {
    return 7;
  }

  /**
   * Empties the raw vectors of the scalar-quantized format's flat storage, as Lucene's own
   * TestLucene104ScalarQuantizedVectorsFormat does. Since Lucene 10.4; not marked @Override, so
   * that it compiles against earlier releases.
   */
  protected void simulateEmptyRawVectors(Directory dir) throws Exception {
    for (String file : dir.listAll()) {
      if (file.endsWith(".vec")) {
        replaceWithEmptyVectorFile(dir, file);
      } else if (file.endsWith(".vemf")) {
        updateVectorMetadataFile(dir, file);
      }
    }
  }

  /** Replaces a raw vector file with an empty one that has a valid header and footer. */
  private static void replaceWithEmptyVectorFile(Directory dir, String fileName) throws Exception {
    byte[] indexHeader;
    try (IndexInput in = dir.openInput(fileName, IOContext.DEFAULT)) {
      indexHeader = CodecUtil.readIndexHeader(in);
    }
    dir.deleteFile(fileName);
    try (IndexOutput out = dir.createOutput(fileName, IOContext.DEFAULT)) {
      out.writeBytes(indexHeader, 0, indexHeader.length);
      CodecUtil.writeFooter(out);
    }
  }

  /** Rewrites the flat vectors' metadata to describe no stored vectors. */
  private static void updateVectorMetadataFile(Directory dir, String fileName) throws Exception {
    byte[] indexHeader;
    int fieldNumber, vectorEncoding, vectorSimilarityFunction, dimension;
    long vectorStartPos;
    try (IndexInput in = dir.openInput(fileName, IOContext.DEFAULT)) {
      indexHeader = CodecUtil.readIndexHeader(in);
      fieldNumber = in.readInt();
      vectorEncoding = in.readInt();
      vectorSimilarityFunction = in.readInt();
      vectorStartPos = in.readVLong();
      in.readVLong(); // the original vector length
      dimension = in.readVInt();
    }
    dir.deleteFile(fileName);
    try (IndexOutput out = dir.createOutput(fileName, IOContext.DEFAULT)) {
      out.writeBytes(indexHeader, 0, indexHeader.length);
      out.writeInt(fieldNumber);
      out.writeInt(vectorEncoding);
      out.writeInt(vectorSimilarityFunction);
      out.writeVLong(vectorStartPos);
      out.writeVLong(0); // no vector data
      out.writeVInt(dimension);
      out.writeInt(0); // no vectors
      // Lucene99FlatVectorsFormat.DIRECT_MONOTONIC_BLOCK_SHIFT, which is package-private.
      OrdToDocDISIReaderConfiguration.writeStoredMeta(16, out, null, 0, 0, null);
      out.writeInt(-1); // end of fields
      CodecUtil.writeFooter(out);
    }
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
        var topDocs = TestLuceneCompat.searchNearestVectors(r, F, queryVector, 2, null, 10);
        assertTrue("Should return at least one result", topDocs.scoreDocs.length > 0);
        assertTrue("Scores should be non-negative", topDocs.scoreDocs[0].score >= 0);
      }
    }
  }
}

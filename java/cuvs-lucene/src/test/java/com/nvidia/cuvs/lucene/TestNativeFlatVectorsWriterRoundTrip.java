/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.generateDataset;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.UUID;
import java.util.function.IntFunction;
import org.apache.commons.io.FileUtils;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/**
 * Positive round-trip check for {@link NativeFlatVectorsWriter}: builds a single-segment index
 * with {@code AcceleratedHNSWParams.numInputVectors} set (native flat buffering) and reads the
 * {@code .vec}/{@code .vemf} files back through the stock {@code Lucene99FlatVectorsReader},
 * asserting every vector round-trips byte-exact.
 *
 * <p>This is the check called for by {@link NativeFlatVectorsWriter}'s "on a Lucene upgrade" class
 * javadoc: it confirms the hand-transcribed format is still readable by Lucene's real reader, and
 * should pass on every {@code lucene-core} version bump.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestNativeFlatVectorsWriterRoundTrip extends LuceneTestCase {

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

  @Test
  public void vectorsRoundTripThroughStockLucene99FlatVectorsReader() throws Exception {
    int numDocs = 500;
    int dimension = 32;
    float[][] dataset = generateDataset(random, numDocs, dimension);

    AcceleratedHNSWParams params =
        new AcceleratedHNSWParams.Builder().withNumInputVectors(numDocs).build();
    Codec codec = new Lucene101AcceleratedHNSWCodec(params);

    // Force everything into a single unsorted, unmerged flush: native flat buffering requires
    // numInputVectors to equal the exact number of vectors landing in that one flush.
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

    try (Directory dir = FSDirectory.open(indexDirPath);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      assertEquals("expected a single native-flat-buffered segment", 1, reader.leaves().size());
      LeafReader leafReader = reader.leaves().get(0).reader();
      assertFieldRoundTrips(leafReader, VECTOR_FIELD, dimension, numDocs, docId -> dataset[docId]);
    }
  }

  /**
   * Cardinality (8000) exceeds {@code IndexedDISI.MAX_ARRAY_LENGTH} (4095), so {@code
   * OrdToDocDISIReaderConfiguration} picks the DENSE (bitset) {@code docsWithField} encoding for
   * this field.
   */
  @Test
  public void vectorsRoundTripWithDenseDocsWithFieldEncoding() throws Exception {
    roundTripPartialField(10000, 8000, 32);
  }

  /**
   * Cardinality (800) stays under {@code IndexedDISI.MAX_ARRAY_LENGTH} (4095), so {@code
   * OrdToDocDISIReaderConfiguration} picks the SPARSE (array) {@code docsWithField} encoding
   * instead.
   */
  @Test
  public void vectorsRoundTripWithSparseDocsWithFieldEncoding() throws Exception {
    roundTripPartialField(1000, 800, 32);
  }

  /**
   * Two vector fields written to the same segment, exercising the boundary between consecutive
   * per-field records in {@code .vemf}: the second field's header must be found where the first
   * field's record actually ends.
   */
  @Test
  public void vectorsRoundTripAcrossMultipleFieldsInSameSegment() throws Exception {
    int numDocs = 500;
    int dimensionA = 32;
    int dimensionB = 16;
    String fieldB = "vector_field_two";
    float[][] datasetA = generateDataset(random, numDocs, dimensionA);
    float[][] datasetB = generateDataset(random, numDocs, dimensionB);

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
        document.add(new KnnFloatVectorField(VECTOR_FIELD, datasetA[i], EUCLIDEAN));
        document.add(new KnnFloatVectorField(fieldB, datasetB[i], EUCLIDEAN));
        writer.addDocument(document);
      }
      writer.commit();
    }

    try (Directory dir = FSDirectory.open(indexDirPath);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      assertEquals("expected a single native-flat-buffered segment", 1, reader.leaves().size());
      LeafReader leafReader = reader.leaves().get(0).reader();
      assertFieldRoundTrips(
          leafReader, VECTOR_FIELD, dimensionA, numDocs, docId -> datasetA[docId]);
      assertFieldRoundTrips(leafReader, fieldB, dimensionB, numDocs, docId -> datasetB[docId]);
    }
  }

  /**
   * Builds a segment of {@code numDocs} documents where only a random {@code numDocsWithVector}
   * of them carry {@code VECTOR_FIELD}, then verifies the round trip.
   */
  private void roundTripPartialField(int numDocs, int numDocsWithVector, int dimension)
      throws Exception {
    Set<Integer> docsWithVector = randomDocSubset(numDocs, numDocsWithVector);
    float[][] dataset = generateDataset(random, numDocsWithVector, dimension);

    AcceleratedHNSWParams params =
        new AcceleratedHNSWParams.Builder().withNumInputVectors(numDocsWithVector).build();
    Codec codec = new Lucene101AcceleratedHNSWCodec(params);

    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(codec)
            .setUseCompoundFile(false)
            .setMaxBufferedDocs(numDocs + 1)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH)
            .setMergePolicy(NoMergePolicy.INSTANCE);

    Map<Integer, Integer> docIdToVectorIndex = new HashMap<>();
    try (Directory dir = FSDirectory.open(indexDirPath);
        IndexWriter writer = new IndexWriter(dir, config)) {
      int vectorIndex = 0;
      for (int i = 0; i < numDocs; i++) {
        Document document = new Document();
        document.add(new StringField(ID_FIELD, Integer.toString(i), Field.Store.YES));
        if (docsWithVector.contains(i)) {
          document.add(new KnnFloatVectorField(VECTOR_FIELD, dataset[vectorIndex], EUCLIDEAN));
          docIdToVectorIndex.put(i, vectorIndex);
          vectorIndex++;
        }
        writer.addDocument(document);
      }
      writer.commit();
    }

    try (Directory dir = FSDirectory.open(indexDirPath);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      assertEquals("expected a single native-flat-buffered segment", 1, reader.leaves().size());
      LeafReader leafReader = reader.leaves().get(0).reader();
      assertFieldRoundTrips(
          leafReader,
          VECTOR_FIELD,
          dimension,
          numDocsWithVector,
          docId -> dataset[docIdToVectorIndex.get(docId)]);
    }
  }

  /** Picks {@code count} distinct doc ids out of {@code [0, numDocs)}. */
  private Set<Integer> randomDocSubset(int numDocs, int count) {
    List<Integer> allDocs = new ArrayList<>(numDocs);
    for (int i = 0; i < numDocs; i++) {
      allDocs.add(i);
    }
    Collections.shuffle(allDocs, random);
    return new HashSet<>(allDocs.subList(0, count));
  }

  private void assertFieldRoundTrips(
      LeafReader leafReader,
      String fieldName,
      int dimension,
      int expectedCount,
      IntFunction<float[]> expectedVectorForDocId)
      throws IOException {
    FloatVectorValues values = leafReader.getFloatVectorValues(fieldName);
    assertNotNull(values);
    assertEquals(expectedCount, values.size());
    assertEquals(dimension, values.dimension());

    int seen = 0;
    KnnVectorValues.DocIndexIterator it = values.iterator();
    for (int doc = it.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = it.nextDoc()) {
      String id = leafReader.storedFields().document(doc).get(ID_FIELD);
      float[] roundTripped = values.vectorValue(it.index());
      assertArrayEquals(
          "vector for field="
              + fieldName
              + " id="
              + id
              + " did not round-trip byte-exact through the stock Lucene99FlatVectorsReader",
          expectedVectorForDocId.apply(Integer.parseInt(id)),
          roundTripped,
          0f);
      seen++;
    }
    assertEquals("did not visit every vector for field=" + fieldName, expectedCount, seen);
  }
}

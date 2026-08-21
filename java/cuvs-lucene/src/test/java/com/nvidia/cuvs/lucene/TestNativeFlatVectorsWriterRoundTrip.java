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
 * <p>This is the check called for by {@link NativeFlatVectorsWriter}'s "on a Lucene upgrade"
 * class javadoc: {@link TestNativeFlatVectorsWriterFormatConstants} only catches a major-version
 * move, but this test is what actually confirms the hand-transcribed format is still readable by
 * Lucene's real reader, and should pass on every {@code lucene-core} version bump.
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
      FloatVectorValues values = leafReader.getFloatVectorValues(VECTOR_FIELD);
      assertNotNull(values);
      assertEquals(numDocs, values.size());
      assertEquals(dimension, values.dimension());

      int seen = 0;
      KnnVectorValues.DocIndexIterator it = values.iterator();
      for (int doc = it.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = it.nextDoc()) {
        String id = leafReader.storedFields().document(doc).get(ID_FIELD);
        float[] roundTripped = values.vectorValue(it.index());
        assertArrayEquals(
            "vector for id="
                + id
                + " did not round-trip byte-exact through the stock"
                + " Lucene99FlatVectorsReader",
            dataset[Integer.parseInt(id)],
            roundTripped,
            0f);
        seen++;
      }
      assertEquals("did not visit every vector", numDocs, seen);
    }
  }
}

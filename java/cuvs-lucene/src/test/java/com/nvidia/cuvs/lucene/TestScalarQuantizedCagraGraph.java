/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.assertIsSupported;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;
import static org.junit.Assume.assumeTrue;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.stream.IntStream;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.InfoStream;
import org.junit.Test;

/** Checks the GPU-built scalar graph with CPU HNSW search against exact Euclidean neighbors. */
public class TestScalarQuantizedCagraGraph extends LuceneTestCase {
  private static final String VECTOR_FIELD = "vector";
  private static final String ID_FIELD = "id";
  private static final int VECTOR_COUNT = 512;
  private static final int DIMENSIONS = 128;
  private static final int TOP_K = 10;
  private static final int MIN_EXACT_NEIGHBORS = 8;

  @Test
  public void testGpuBuiltScalarHnswRetainsRecallAcrossMerge() throws Exception {
    requireGpuWhenSelected();

    float[][] vectors = vectorsWithNegativeAndMixedSignDimensions();
    RecordingInfoStream buildLog = new RecordingInfoStream();
    AcceleratedHNSWParams params =
        new AcceleratedHNSWParams.Builder()
            .withStrategy(AcceleratedHNSWParams.Strategy.CUSTOM)
            .withGraphDegree(32)
            .withIntermediateGraphDegree(64)
            .withHNSWLayer(1)
            .build();
    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(
                TestUtil.alwaysKnnVectorsFormat(
                    new LuceneAcceleratedHNSWScalarQuantizedVectorsFormat(params)))
            .setMaxBufferedDocs(VECTOR_COUNT + 1)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH)
            .setInfoStream(buildLog);

    try (Directory directory = newDirectory()) {
      try (IndexWriter writer = new IndexWriter(directory, config)) {
        for (int id = 0; id < vectors.length; id++) {
          Document document = new Document();
          document.add(new StringField(ID_FIELD, Integer.toString(id), Field.Store.YES));
          document.add(new KnnFloatVectorField(VECTOR_FIELD, vectors[id], EUCLIDEAN));
          writer.addDocument(document);
          if (id == VECTOR_COUNT / 2 - 1) {
            writer.commit();
          }
        }
        writer.commit();

        try (DirectoryReader reader = DirectoryReader.open(directory)) {
          assertEquals(2, reader.leaves().size());
          assertHnswRecallAgainstExactNeighbors(reader, vectors);
        }

        long gpuBuildsBeforeMerge = buildLog.gpuWriterOpenCount();
        assertTrue("No scalar GPU writer opened: " + buildLog.messages, gpuBuildsBeforeMerge >= 2);

        writer.forceMerge(1);
        writer.commit();
        assertTrue(
            "The merge did not open a scalar GPU writer: " + buildLog.messages,
            buildLog.gpuWriterOpenCount() > gpuBuildsBeforeMerge);
      }

      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        assertEquals(1, reader.leaves().size());
        assertHnswRecallAgainstExactNeighbors(reader, vectors);
      }
    }
  }

  private static void requireGpuWhenSelected() {
    if (Boolean.getBoolean("cuvs.lucene.tests.requireGpu")) {
      assertIsSupported();
    } else {
      assumeTrue("cuVS is not supported", isSupported());
    }
  }

  private static void assertHnswRecallAgainstExactNeighbors(
      DirectoryReader reader, float[][] vectors) throws Exception {
    IndexSearcher searcher = new IndexSearcher(reader);
    for (int queryId : new int[] {0, 240, 272, VECTOR_COUNT - 1}) {
      var results =
          searcher.search(new KnnFloatVectorQuery(VECTOR_FIELD, vectors[queryId], TOP_K), TOP_K);
      List<Integer> actual = new ArrayList<>();
      for (var hit : results.scoreDocs) {
        actual.add(Integer.parseInt(searcher.storedFields().document(hit.doc).get(ID_FIELD)));
      }

      assertEquals("Query " + queryId, TOP_K, actual.size());
      assertEquals("Query " + queryId, queryId, actual.get(0).intValue());
      assertEquals(
          "Query " + queryId + " returned duplicates", TOP_K, new HashSet<>(actual).size());

      List<Integer> exact =
          IntStream.range(0, vectors.length)
              .boxed()
              .sorted(
                  Comparator.comparingDouble(
                          (Integer id) -> squaredDistance(vectors[queryId], vectors[id]))
                      .thenComparingInt(Integer::intValue))
              .limit(TOP_K)
              .toList();
      Set<Integer> expected = new HashSet<>(exact);
      long overlap = actual.stream().filter(expected::contains).count();
      assertTrue(
          "Query " + queryId + " found " + overlap + "/" + TOP_K + " exact neighbors: " + actual,
          overlap >= MIN_EXACT_NEIGHBORS);
    }
  }

  private static double squaredDistance(float[] left, float[] right) {
    double distance = 0;
    for (int dimension = 0; dimension < left.length; dimension++) {
      double difference = left[dimension] - right[dimension];
      distance += difference * difference;
    }
    return distance;
  }

  private static float[][] vectorsWithNegativeAndMixedSignDimensions() {
    float[][] vectors = new float[VECTOR_COUNT][DIMENSIONS];
    for (int id = 0; id < VECTOR_COUNT; id++) {
      int x = id % 32;
      int y = id / 32;
      for (int dimension = 0; dimension < DIMENSIONS; dimension++) {
        vectors[id][dimension] =
            switch (dimension % 4) {
              case 0 -> -20.0f + 0.4f * x + 0.006f * y; // all negative
              case 1 -> -7.5f + 0.9f * y + 0.002f * x; // crosses zero
              case 2 -> 4.0f + 0.25f * x + 0.003f * y; // all positive
              default -> -6.0f + 0.21f * (x + y); // crosses zero
            };
      }
    }
    return vectors;
  }

  private static final class RecordingInfoStream extends InfoStream {
    private static final String GPU_WRITER_OPENED =
        "Lucene99AcceleratedHNSWQuantizedVectorsWriter opened";
    private final List<String> messages = new CopyOnWriteArrayList<>();

    @Override
    public void message(String component, String message) {
      messages.add(component + ": " + message);
    }

    @Override
    public boolean isEnabled(String component) {
      return true;
    }

    @Override
    public void close() {}

    private long gpuWriterOpenCount() {
      return messages.stream().filter(message -> message.contains(GPU_WRITER_OPENED)).count();
    }
  }
}

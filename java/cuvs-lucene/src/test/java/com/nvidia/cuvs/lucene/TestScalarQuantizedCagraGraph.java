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
  private static final int GRID_COLUMNS = 32;
  private static final int INITIAL_SEGMENT_COUNT = 2;
  private static final int DOCUMENTS_PER_SEGMENT = VECTOR_COUNT / INITIAL_SEGMENT_COUNT;
  private static final int MERGED_SEGMENT_COUNT = 1;
  private static final int DIMENSIONS = 128;
  private static final int DIMENSION_PATTERN_COUNT = 4;
  private static final int TOP_K = 10;
  private static final int MIN_EXACT_NEIGHBORS = 8;
  private static final int CAGRA_GRAPH_DEGREE = 32;
  private static final int CAGRA_INTERMEDIATE_GRAPH_DEGREE = 64;
  private static final int HNSW_LAYER_COUNT = 1;

  @Test
  public void testGpuBuiltScalarHnswRetainsRecallAcrossMerge() throws Exception {
    requireGpuWhenSelected();

    float[][] vectors = vectorsWithNegativeAndMixedSignDimensions();
    RecordingInfoStream buildLog = new RecordingInfoStream();
    AcceleratedHNSWParams params =
        new AcceleratedHNSWParams.Builder()
            .withStrategy(AcceleratedHNSWParams.Strategy.CUSTOM)
            .withGraphDegree(CAGRA_GRAPH_DEGREE)
            .withIntermediateGraphDegree(CAGRA_INTERMEDIATE_GRAPH_DEGREE)
            .withHNSWLayer(HNSW_LAYER_COUNT)
            .build();
    int maxBufferedDocsWithoutAutomaticFlush = VECTOR_COUNT + 1;
    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(
                TestUtil.alwaysKnnVectorsFormat(
                    new LuceneAcceleratedHNSWScalarQuantizedVectorsFormat(params)))
            .setMaxBufferedDocs(maxBufferedDocsWithoutAutomaticFlush)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH)
            .setInfoStream(buildLog);

    try (Directory directory = newDirectory()) {
      try (IndexWriter writer = new IndexWriter(directory, config)) {
        for (int id = 0; id < vectors.length; id++) {
          Document document = new Document();
          document.add(new StringField(ID_FIELD, Integer.toString(id), Field.Store.YES));
          document.add(new KnnFloatVectorField(VECTOR_FIELD, vectors[id], EUCLIDEAN));
          writer.addDocument(document);
          if (id == DOCUMENTS_PER_SEGMENT - 1) {
            writer.commit();
          }
        }
        writer.commit();

        try (DirectoryReader reader = DirectoryReader.open(directory)) {
          assertEquals(INITIAL_SEGMENT_COUNT, reader.leaves().size());
          assertHnswRecallAgainstExactNeighbors(reader, vectors);
        }

        long gpuBuildsBeforeMerge = buildLog.gpuWriterOpenCount();
        assertTrue(
            "Insufficient scalar GPU writer openings for initial segments: " + buildLog.messages,
            gpuBuildsBeforeMerge >= INITIAL_SEGMENT_COUNT);

        writer.forceMerge(MERGED_SEGMENT_COUNT);
        writer.commit();
        assertTrue(
            "The merge did not open a scalar GPU writer: " + buildLog.messages,
            buildLog.gpuWriterOpenCount() > gpuBuildsBeforeMerge);
      }

      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        assertEquals(MERGED_SEGMENT_COUNT, reader.leaves().size());
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
    int centerColumn = GRID_COLUMNS / 2;
    int lastRowOfFirstSegmentCenter = DOCUMENTS_PER_SEGMENT - GRID_COLUMNS + centerColumn;
    int firstRowOfSecondSegmentCenter = DOCUMENTS_PER_SEGMENT + centerColumn;
    // Probe opposite grid corners and adjacent rows across the segment split.
    for (int queryId :
        new int[] {
          0, lastRowOfFirstSegmentCenter, firstRowOfSecondSegmentCenter, VECTOR_COUNT - 1
        }) {
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
      int column = id % GRID_COLUMNS;
      int row = id / GRID_COLUMNS;
      for (int dimension = 0; dimension < DIMENSIONS; dimension++) {
        vectors[id][dimension] =
            switch (dimension % DIMENSION_PATTERN_COUNT) {
              case 0 -> -20.0f + 0.4f * column + 0.006f * row; // all negative
              case 1 -> -7.5f + 0.9f * row + 0.002f * column; // crosses zero
              case 2 -> 4.0f + 0.25f * column + 0.003f * row; // all positive
              default -> -6.0f + 0.21f * (column + row); // crosses zero
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

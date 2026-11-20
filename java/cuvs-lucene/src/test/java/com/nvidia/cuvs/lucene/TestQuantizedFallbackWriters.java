/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.graphOf;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.closeCuVSResourcesInstance;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.setCuVSResourcesInstance;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import java.io.IOException;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.logging.Handler;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.junit.AfterClass;
import org.junit.BeforeClass;

/**
 * Checks that the accelerated HNSW codecs build the graph on the CPU when cuVS is not available,
 * and that they read back what their CPU fallback wrote.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestQuantizedFallbackWriters extends LuceneTestCase {

  private static final int NUM_DOCS = 300;
  // Enough vectors for Lucene to build a graph, split into segments. Since Lucene 10.4, which
  // builds one only over more than 100 * ln(n) vectors, the segments are too small for one, so
  // only the merge builds it; earlier releases also build one when they flush each segment.
  private static final int MERGED_DOCS = 1000;
  private static final int DOCS_PER_SEGMENT = 250;
  private static final int DIMENSIONS = 16;
  private static final int TOP_K = 10;
  // The sizes of the source segments of the merges in checkMergesGraphsWithOtherM: enough vectors
  // for Lucene 10.4+ to build a graph when it flushes a segment, which it does only over more than
  // 100 * ln(n) vectors. The first is larger, so that Lucene would start the merged graph from it;
  // where the second records the fallback's maxConn, only the first records another M, as where
  // the large merged segments of an index were written by an earlier release.
  private static final int[] OTHER_M_SEGMENT_DOCS = {1000, 700};

  /** Creates a codec that writes with an accelerated HNSW format, and names it for messages. */
  private record CodecFactory(String name, Function<AcceleratedHNSWParams, Codec> codecWith) {}

  /**
   * The accelerated HNSW codecs, and codecs with each accelerated HNSW format set per field, as an
   * application that does not use the codecs has them.
   */
  private static final List<CodecFactory> CODEC_FACTORIES =
      List.of(
          new CodecFactory("acceleratedHNSW", CuVSCodecs::acceleratedHNSW),
          new CodecFactory(
              "acceleratedHNSWScalarQuantized", CuVSCodecs::acceleratedHNSWScalarQuantized),
          new CodecFactory(
              "acceleratedHNSWBinaryQuantized", CuVSCodecs::acceleratedHNSWBinaryQuantized),
          new CodecFactory(
              "acceleratedHNSWFormat per field",
              params -> TestUtil.alwaysKnnVectorsFormat(CuVSCodecs.acceleratedHNSWFormat(params))),
          new CodecFactory(
              "acceleratedHNSWScalarQuantizedFormat per field",
              params ->
                  TestUtil.alwaysKnnVectorsFormat(
                      CuVSCodecs.acceleratedHNSWScalarQuantizedFormat(params))),
          new CodecFactory(
              "acceleratedHNSWBinaryQuantizedFormat per field",
              params ->
                  TestUtil.alwaysKnnVectorsFormat(
                      CuVSCodecs.acceleratedHNSWBinaryQuantizedFormat(params))));

  @BeforeClass
  public static void beforeClass() {
    // Where cuVS is unavailable (no GPU, JDK 21) the writers fall back by themselves. Elsewhere,
    // make isSupported() false on this thread, like on a host without a GPU, after closing the
    // resources isSupported() just created, so that they do not leak.
    if (isSupported()) {
      closeCuVSResourcesInstance();
      setCuVSResourcesInstance(null);
    }
  }

  @AfterClass
  public static void afterClass() {
    // Forget the null set above; the next use on this thread creates resources again, if any.
    closeCuVSResourcesInstance();
  }

  public void testFallbackWriters() throws Exception {
    AcceleratedHNSWParams params =
        new AcceleratedHNSWParams.Builder().withMaxConn(16).withBeamWidth(100).build();
    for (Codec codec :
        List.of(
            CuVSCodecs.acceleratedHNSW(params),
            CuVSCodecs.acceleratedHNSWScalarQuantized(params),
            CuVSCodecs.acceleratedHNSWBinaryQuantized(params))) {
      checkWritesAndReads(codec);
    }
  }

  /**
   * Merges segments with the CPU fallback into one large enough for a graph. Since Lucene 10.5, the
   * fallback builds the graph of a merged segment from the flat vectors it has just written, which
   * it reads back with the flat format the codec passes it.
   */
  public void testFallbackWritersMerge() throws Exception {
    AcceleratedHNSWParams params =
        new AcceleratedHNSWParams.Builder().withMaxConn(16).withBeamWidth(100).build();
    for (Codec codec :
        List.of(
            CuVSCodecs.acceleratedHNSW(params),
            CuVSCodecs.acceleratedHNSWScalarQuantized(params),
            CuVSCodecs.acceleratedHNSWBinaryQuantized(params))) {
      float[][] vectors = randomVectors(new Random(random().nextLong()), MERGED_DOCS);
      try (Directory dir = newDirectory()) {
        int fallbacks = writeIndex(dir, codec, vectors, DOCS_PER_SEGMENT, true);
        assertTrue(codec.getName() + " did not fall back to the CPU", fallbacks > 0);
        try (DirectoryReader reader = DirectoryReader.open(dir)) {
          assertEquals(codec.getName(), 1, reader.leaves().size());
          assertEquals(
              codec.getName() + ": the merged segment should have a graph over every vector",
              MERGED_DOCS,
              graphOf(reader.leaves().get(0).reader(), "vector").size());
        }
        assertVectorsFindThemselves(dir, codec.getName(), vectors);
      }
    }
  }

  /**
   * Merges, with the CPU fallback, segments whose graphs record a smaller M than the fallback's
   * maxConn, as segments written by cuvs-lucene 26.10 and earlier can. Since Lucene 10.4, Lucene's
   * writer merging from such graphs fails with "No growth is allowed"; see {@link
   * FallbackHnswVectorsWriter}. cuvs-lucene no longer records such an M, so the source segments
   * here are written with a smaller maxConn.
   */
  public void testFallbackWritersMergeGraphsWithSmallerM() throws Exception {
    checkMergesGraphsWithOtherM(4, 32, false, false);
  }

  /**
   * Like {@link #testFallbackWritersMergeGraphsWithSmallerM}, but merges a segment whose graph
   * records a smaller M with one whose graph records the fallback's maxConn, as after an upgrade
   * from cuvs-lucene 26.10 or earlier.
   */
  public void testFallbackWritersMergeGraphsWithSmallerAndSameM() throws Exception {
    checkMergesGraphsWithOtherM(4, 32, false, true);
  }

  /**
   * Merges, with the CPU fallback, segments whose graphs record a larger M than the fallback's
   * maxConn, as segments written by cuvs-lucene 26.10 and earlier with a large {@code CUSTOM}
   * graph degree can. Lucene's writer merging from such graphs fails with "No growth is allowed" up
   * to Lucene 10.3, and since Lucene 10.4 can write nodes with more neighbors than a reader of the
   * merged segment reads; see {@link FallbackHnswVectorsWriter}. The source segments here are
   * written with a larger maxConn.
   */
  public void testFallbackWritersMergeGraphsWithLargerM() throws Exception {
    checkMergesGraphsWithOtherM(32, 16, true, false);
  }

  /**
   * Like {@link #testFallbackWritersMergeGraphsWithLargerM}, but merges a segment whose graph
   * records a larger M with one whose graph records the fallback's maxConn.
   */
  public void testFallbackWritersMergeGraphsWithLargerAndSameM() throws Exception {
    checkMergesGraphsWithOtherM(32, 16, true, true);
  }

  /**
   * Writes two segments with {@code sourceMaxConn}, or the second with {@code maxConn} if {@code
   * secondWithMaxConn}, and merges them with {@code maxConn}, with each of {@link
   * #CODEC_FACTORIES}. Lucene 10.4 only merges from the graph of a segment whose vectors reader is
   * not per field, so with the formats set per field, it checks there only that the merge is right.
   *
   * @param wideSourceNodes whether to check that the graphs with {@code sourceMaxConn} have nodes
   *     with more neighbors than a graph with {@code maxConn} holds, which the merge must not keep
   */
  private void checkMergesGraphsWithOtherM(
      int sourceMaxConn, int maxConn, boolean wideSourceNodes, boolean secondWithMaxConn)
      throws Exception {
    AcceleratedHNSWParams params =
        new AcceleratedHNSWParams.Builder().withMaxConn(maxConn).withBeamWidth(100).build();
    AcceleratedHNSWParams sourceParams =
        new AcceleratedHNSWParams.Builder().withMaxConn(sourceMaxConn).withBeamWidth(100).build();
    for (CodecFactory factory : CODEC_FACTORIES) {
      String name = factory.name();
      // High-dimensional random vectors pass HNSW's diversity check often enough to fill a node's
      // neighbors.
      float[][] vectors =
          randomVectors(
              new Random(random().nextLong()),
              OTHER_M_SEGMENT_DOCS[0] + OTHER_M_SEGMENT_DOCS[1],
              64);
      try (Directory dir = newDirectory()) {
        // One segment per writer, so that each can have its own maxConn.
        for (int segment = 0; segment < 2; segment++) {
          int segmentMaxConn = secondWithMaxConn && segment == 1 ? maxConn : sourceMaxConn;
          IndexWriterConfig sourceConfig =
              new IndexWriterConfig()
                  .setCodec(
                      factory.codecWith().apply(segmentMaxConn == maxConn ? params : sourceParams))
                  .setMergePolicy(NoMergePolicy.INSTANCE);
          try (IndexWriter writer = new IndexWriter(dir, sourceConfig)) {
            int from = segment == 0 ? 0 : OTHER_M_SEGMENT_DOCS[0];
            addDocuments(writer, vectors, from, from + OTHER_M_SEGMENT_DOCS[segment]);
          }
        }
        try (DirectoryReader reader = DirectoryReader.open(dir)) {
          assertEquals(name, 2, reader.leaves().size());
          int widest = 0;
          for (int segment = 0; segment < 2; segment++) {
            HnswGraph graph = graphOf(reader.leaves().get(segment).reader(), "vector");
            assertEquals(name, OTHER_M_SEGMENT_DOCS[segment], graph.size());
            if (secondWithMaxConn && segment == 1) {
              assertEquals(name, maxConn, graph.maxConn());
            } else {
              assertEquals(name, sourceMaxConn, graph.maxConn());
              widest = Math.max(widest, mostNeighbors(graph, 0));
            }
          }
          if (wideSourceNodes) {
            assertTrue(
                name + ": the source graphs have no node with more than 2 * maxConn neighbors",
                widest > 2 * maxConn);
          }
        }

        IndexWriterConfig mergeConfig =
            new IndexWriterConfig()
                .setCodec(factory.codecWith().apply(params))
                .setMergePolicy(new TieredMergePolicy())
                // Merges on this thread, where cuVS is unavailable.
                .setMergeScheduler(new SerialMergeScheduler());
        int fallbacks;
        try (IndexWriter writer = new IndexWriter(dir, mergeConfig)) {
          fallbacks = countFallbacks(() -> writer.forceMerge(1));
        }
        assertTrue(name + " did not merge on the CPU", fallbacks > 0);
        try (DirectoryReader reader = DirectoryReader.open(dir)) {
          assertEquals(name, 1, reader.leaves().size());
          HnswGraph graph = graphOf(reader.leaves().get(0).reader(), "vector");
          assertEquals(name, vectors.length, graph.size());
          assertEquals(name, maxConn, graph.maxConn());
          // Lucene's builder gives a node up to 2 * M neighbors on level 0 and up to M above it. A
          // reader fails on a node with more than 2 * M on any level, and a later merge starting
          // from this graph fails on a node above level 0 with more than M.
          for (int level = 0; level < graph.numLevels(); level++) {
            int limit = level == 0 ? 2 * maxConn : maxConn;
            int most = mostNeighbors(graph, level);
            assertTrue(
                name
                    + ": a node on level "
                    + level
                    + " of the merged graph has "
                    + most
                    + " neighbors, more than "
                    + limit,
                most <= limit);
          }
        }
        assertVectorsFindThemselves(dir, name, vectors);
      }
    }
  }

  /** Returns the most neighbors that a node of {@code graph} has on {@code level}. */
  private static int mostNeighbors(HnswGraph graph, int level) throws IOException {
    int most = 0;
    HnswGraph.NodesIterator nodes = graph.getNodesOnLevel(level);
    while (nodes.hasNext()) {
      graph.seek(level, nodes.nextInt());
      most = Math.max(most, graph.neighborCount());
    }
    return most;
  }

  private void checkWritesAndReads(Codec codec) throws Exception {
    float[][] vectors = randomVectors(new Random(random().nextLong()), NUM_DOCS);
    try (Directory dir = newDirectory()) {
      int fallbacks = writeIndex(dir, codec, vectors, IndexWriterConfig.DISABLE_AUTO_FLUSH, false);
      assertTrue(codec.getName() + " did not fall back to the CPU", fallbacks > 0);
      assertVectorsFindThemselves(dir, codec.getName(), vectors);
    }
  }

  private static float[][] randomVectors(Random random, int count) {
    return randomVectors(random, count, DIMENSIONS);
  }

  private static float[][] randomVectors(Random random, int count, int dimensions) {
    float[][] vectors = new float[count][dimensions];
    for (float[] vector : vectors) {
      for (int d = 0; d < dimensions; d++) {
        vector[d] = random.nextFloat();
      }
    }
    return vectors;
  }

  /**
   * Indexes one document per vector, with its position as the stored "id", flushing every {@code
   * maxBufferedDocs} documents and then merging into one segment if {@code forceMerge}, which must
   * fall back to the CPU too, and returns how many times the codec logged that it fell back.
   */
  private static int writeIndex(
      Directory dir, Codec codec, float[][] vectors, int maxBufferedDocs, boolean forceMerge)
      throws Exception {
    // Flushes and merges on this thread, where cuVS is unavailable.
    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(codec)
            .setUseCompoundFile(false)
            .setMaxBufferedDocs(maxBufferedDocs)
            .setMergePolicy(forceMerge ? new TieredMergePolicy() : NoMergePolicy.INSTANCE)
            .setMergeScheduler(new SerialMergeScheduler());
    // Counts the fallbacks of the flush on close too.
    return countFallbacks(
        () -> {
          try (IndexWriter writer = new IndexWriter(dir, config)) {
            addDocuments(writer, vectors);
            if (forceMerge) {
              // Flush the last segment first, so that the fallbacks counted next come from the
              // merge.
              writer.flush();
              int mergeFallbacks = countFallbacks(() -> writer.forceMerge(1));
              assertTrue(codec.getName() + " did not merge on the CPU", mergeFallbacks > 0);
            }
          }
        });
  }

  /** Something done with an index. */
  private interface IndexAction {
    void run() throws Exception;
  }

  /** Runs {@code action} and returns how many times a codec logged that it fell back to the CPU. */
  private static int countFallbacks(IndexAction action) throws Exception {
    AtomicInteger fallbacks = new AtomicInteger();
    Handler countFallbacks =
        new Handler() {
          @Override
          public void publish(LogRecord record) {
            if (record.getMessage() != null && record.getMessage().contains("falling back")) {
              fallbacks.incrementAndGet();
            }
          }

          @Override
          public void flush() {}

          @Override
          public void close() {}
        };
    Logger logger = Logger.getLogger("com.nvidia.cuvs.lucene");
    logger.addHandler(countFallbacks);
    try {
      action.run();
    } finally {
      logger.removeHandler(countFallbacks);
    }
    return fallbacks.get();
  }

  /** Indexes one document per vector, with its position as the stored "id". */
  private static void addDocuments(IndexWriter writer, float[][] vectors) throws Exception {
    addDocuments(writer, vectors, 0, vectors.length);
  }

  /**
   * Indexes one document per vector from {@code from} (inclusive) to {@code to} (exclusive), with
   * its position as the stored "id".
   */
  private static void addDocuments(IndexWriter writer, float[][] vectors, int from, int to)
      throws Exception {
    for (int i = from; i < to; i++) {
      Document doc = new Document();
      doc.add(new StoredField("id", i));
      doc.add(new KnnFloatVectorField("vector", vectors[i], EUCLIDEAN));
      writer.addDocument(doc);
    }
  }

  /** Checks that at least 90% of the vectors are among their own top-k search results. */
  private static void assertVectorsFindThemselves(Directory dir, String name, float[][] vectors)
      throws Exception {
    try (DirectoryReader reader = DirectoryReader.open(dir)) {
      assertEquals(name, vectors.length, reader.numDocs());
      IndexSearcher searcher = new IndexSearcher(reader);
      int found = 0;
      for (int i = 0; i < vectors.length; i++) {
        ScoreDoc[] hits =
            searcher.search(new KnnFloatVectorQuery("vector", vectors[i], TOP_K), TOP_K).scoreDocs;
        for (ScoreDoc hit : hits) {
          if (searcher.storedFields().document(hit.doc).getField("id").numericValue().intValue()
              == i) {
            found++;
            break;
          }
        }
      }
      assertTrue(
          name + ": only " + found + " of " + vectors.length + " vectors found themselves",
          found >= vectors.length * 0.9);
    }
  }
}

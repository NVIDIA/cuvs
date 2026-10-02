/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import java.util.Arrays;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.junit.Test;

/**
 * Verifies the HNSW {@code M} recorded in a segment's metadata, for every segment an
 * accelerated-HNSW writer can produce.
 *
 * <p>{@code M} has two lower bounds. It bounds what the reader accepts: Lucene sizes its arc buffer
 * as {@code M * 2} and asserts that every stored adjacency row fits, so {@code M} must cover {@code
 * ceil(cagraGraphDegree / 2)} of the graph actually built. An {@code M} taken from the configured
 * graph degree can understate the graph, because cuVS is free to build a degree other than the one
 * requested -- and under the HEURISTIC strategy it derives the degree from maxConn and ignores the
 * configured graph degree outright.
 *
 * <p>{@code M} must also not fall below maxConn. On Lucene 10.4+ a CPU merge sizes the merged
 * graph's neighbor arrays from the {@code M} of its largest source segment but fills them up to
 * {@code 2 * maxConn}. cuVS truncates the CAGRA graph degree to {@code dataset_size - 1} for small
 * datasets, so the graph alone would give a 20-vector segment {@code M = 10} with maxConn 16, and a
 * single-vector segment an even smaller one; merging either throws.
 *
 * <p>A non-default maxConn is used throughout. At stock defaults the configured graph degree (64)
 * and the degree cuVS derives from maxConn (2 * 32) coincide, so an {@code M} read from the wrong
 * source would still produce the expected value and go unnoticed.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestSegmentMaxConnConsistency extends LuceneTestCase {

  private static final String FIELD = "f";
  private static final int MAX_CONN = 16;

  /**
   * Every segment -- including a degenerate single-vector one -- must record an M that covers both
   * its own widest adjacency row and maxConn.
   */
  @Test
  public void testRecordedMCoversEachSegmentsGraphAndMaxConn() throws Exception {
    assumeTrue("cuVS not supported", isSupported());

    // Sizes span the interesting cases: the single-vector special path, a segment small enough for
    // cuVS to truncate the degree, and one large enough to keep the derived degree.
    int[] segmentSizes = {1, 20, 3000};

    try (Directory dir = newDirectory()) {
      IndexWriterConfig cfg =
          new IndexWriterConfig()
              .setCodec(
                  CuVSCodecs.acceleratedHNSW(
                      new AcceleratedHNSWParams.Builder()
                          .withStrategy(AcceleratedHNSWParams.Strategy.HEURISTIC)
                          .withMaxConn(MAX_CONN)
                          .build()));
      // Keep the segments separate so each one's metadata can be inspected.
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);

      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int size : segmentSizes) {
          addDocs(w, size);
          w.commit();
        }
      }

      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        assertEquals("expected one segment per size", segmentSizes.length, reader.leaves().size());

        for (LeafReaderContext ctx : reader.leaves()) {
          LeafReader leaf = ctx.reader();
          int size = leaf.getFloatVectorValues(FIELD).size();
          HnswGraph graph = graphOf(leaf);
          int recordedM = graph.maxConn();
          int widestRow = widestAdjacencyRow(graph);

          assertEquals(
              "segment of "
                  + size
                  + " vectors recorded M="
                  + recordedM
                  + " but its widest adjacency row holds "
                  + widestRow
                  + " arcs and maxConn is "
                  + MAX_CONN,
              Math.max(MAX_CONN, Math.ceilDiv(widestRow, 2)),
              recordedM);

          // The reader sizes its arc buffer as M*2 and asserts every arc count fits, so an M that
          // understates the graph corrupts reads regardless of where it came from.
          assertTrue(
              "segment of "
                  + size
                  + " vectors has "
                  + widestRow
                  + " arcs but only M*2="
                  + (recordedM * 2),
              widestRow <= recordedM * 2);
        }
      }
    }
  }

  /**
   * The single-vector path must not fall back to the configured graph degree, which the HEURISTIC
   * strategy does not use.
   */
  @Test
  public void testSingleVectorSegmentDoesNotUseConfiguredGraphDegree() throws Exception {
    assumeTrue("cuVS not supported", isSupported());

    AcceleratedHNSWParams params =
        new AcceleratedHNSWParams.Builder()
            .withStrategy(AcceleratedHNSWParams.Strategy.HEURISTIC)
            .withMaxConn(MAX_CONN)
            .withGraphDegree(256) // ignored under HEURISTIC; must not leak into the metadata
            .build();

    try (Directory dir = newDirectory()) {
      IndexWriterConfig cfg = new IndexWriterConfig().setCodec(CuVSCodecs.acceleratedHNSW(params));
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        addDocs(w, 1);
        w.commit();
      }

      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        LeafReader leaf = getOnlyLeafReader(reader);
        HnswGraph graph = graphOf(leaf);
        assertEquals("single-vector segment must record maxConn as M", MAX_CONN, graph.maxConn());
        assertNotEquals("M leaked from the configured graphDegree", 256 / 2, graph.maxConn());
      }
    }
  }

  /**
   * The widest graph the CUSTOM strategy accepts has degree {@code 2 * maxConn}, the most neighbors
   * HNSW holds per node on level 0. It records M = maxConn and merges on the CPU.
   */
  @Test
  public void testCustomGraphDegreeAtTheLimit() throws Exception {
    assumeTrue("cuVS not supported", isSupported());

    AcceleratedHNSWParams params = customParams(2 * MAX_CONN);
    try (Directory dir = newDirectory()) {
      IndexWriterConfig cfg = new IndexWriterConfig().setCodec(CuVSCodecs.acceleratedHNSW(params));
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        addDocs(w, 3000);
        w.forceMerge(1);
      }

      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        HnswGraph graph = graphOf(getOnlyLeafReader(reader));
        assertEquals(2 * MAX_CONN, widestAdjacencyRow(graph));
        assertEquals(MAX_CONN, graph.maxConn());
      }
    }
    cpuMergeGpuSegments(params, repeat(1000, 2));
  }

  private static AcceleratedHNSWParams customParams(int graphDegree) {
    return new AcceleratedHNSWParams.Builder()
        .withStrategy(AcceleratedHNSWParams.Strategy.CUSTOM)
        .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
        .withIntermediateGraphDegree(2 * graphDegree)
        .withGraphDegree(graphDegree)
        .withMaxConn(MAX_CONN)
        .build();
  }

  /**
   * A CPU merge of GPU-written segments, which the formats' fallback writers and stock Lucene do,
   * must have room for {@code 2 * maxConn} neighbors per node. On Lucene 10.4+ the merged graph is
   * sized from the M recorded by its largest source segment, so merging small segments that
   * recorded their truncated degree failed with "No growth is allowed". Lucene 10.2 and 10.3 size
   * it from the merging writer's own M, so there this test always passes.
   */
  @Test
  public void testCpuMergeOfSmallGpuSegments() throws Exception {
    assumeTrue("cuVS not supported", isSupported());
    cpuMergeGpuSegments(heuristicParams(1), repeat(20, 50));
  }

  /**
   * A single-vector segment's only node has no neighbors, which must be written as an empty
   * neighbor list, as Lucene does. A placeholder neighbor {@code -1} made a CPU merge that includes
   * the segment fail on every Lucene release.
   */
  @Test
  public void testCpuMergeOfSingleVectorGpuSegments() throws Exception {
    assumeTrue("cuVS not supported", isSupported());
    int[] sizes = repeat(20, 50);
    int[] withSingles = Arrays.copyOf(sizes, sizes.length + 10);
    Arrays.fill(withSingles, sizes.length, withSingles.length, 1);
    cpuMergeGpuSegments(heuristicParams(1), withSingles);
  }

  /**
   * HNSW allows maxConn neighbors per node above level 0. Upper layers built with the level-0
   * degree of 2 * maxConn were read fine, but a CPU merge copies them into arrays of maxConn + 1
   * slots and failed with "No growth is allowed" on every Lucene release.
   */
  @Test
  public void testCpuMergeOfMultiLayerGpuSegments() throws Exception {
    assumeTrue("cuVS not supported", isSupported());
    cpuMergeGpuSegments(heuristicParams(3), repeat(1000, 2));
  }

  private static AcceleratedHNSWParams heuristicParams(int hnswLayers) {
    return new AcceleratedHNSWParams.Builder()
        .withStrategy(AcceleratedHNSWParams.Strategy.HEURISTIC)
        .withMaxConn(MAX_CONN)
        .withHNSWLayer(hnswLayers)
        .build();
  }

  private static int[] repeat(int value, int count) {
    int[] values = new int[count];
    Arrays.fill(values, value);
    return values;
  }

  /**
   * Writes one GPU-built segment per size, merges them into one on the CPU with Lucene's own HNSW
   * format, and checks the merged index.
   */
  private void cpuMergeGpuSegments(AcceleratedHNSWParams params, int[] segmentSizes)
      throws Exception {
    // High-dimensional random vectors pass the diversity check often enough to fill a node's
    // neighbor array, which a low-dimensional dataset would not.
    int dimensions = 256;
    // Lucene 10.4+ skips the graph of a segment smaller than about 650 vectors (see
    // Lucene99HnswVectorsFormat.HNSW_GRAPH_THRESHOLD), so the merged segment must be larger for the
    // merge to build one.
    int total = 0;
    for (int size : segmentSizes) {
      total += size;
    }
    assertTrue("merged segment too small to get a graph: " + total, total >= 1000);

    try (Directory dir = newDirectory()) {
      IndexWriterConfig gpuCfg =
          new IndexWriterConfig().setCodec(CuVSCodecs.acceleratedHNSW(params));
      gpuCfg.setMergePolicy(NoMergePolicy.INSTANCE);
      try (IndexWriter w = new IndexWriter(dir, gpuCfg)) {
        for (int size : segmentSizes) {
          addRandomDocs(w, size, dimensions);
          w.commit();
        }
      }

      // Every GPU-built level above 0 must fit HNSW's maxConn neighbors per node.
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        for (LeafReaderContext ctx : reader.leaves()) {
          HnswGraph graph = graphOf(ctx.reader());
          for (int level = 1; level < graph.numLevels(); level++) {
            int widest = widestRowOnLevel(graph, level);
            assertTrue("level " + level + " has a row of " + widest, widest <= MAX_CONN);
          }
        }
      }

      IndexWriterConfig cpuCfg =
          new IndexWriterConfig()
              .setCodec(
                  TestUtil.alwaysKnnVectorsFormat(new Lucene99HnswVectorsFormat(MAX_CONN, 100)));
      try (IndexWriter w = new IndexWriter(dir, cpuCfg)) {
        w.forceMerge(1);
      }

      TestUtil.checkIndex(dir);
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        LeafReader leaf = getOnlyLeafReader(reader);
        assertEquals(total, graphOf(leaf).size());
      }
    }
  }

  private static void addRandomDocs(IndexWriter w, int count, int dimensions) throws Exception {
    for (int i = 0; i < count; i++) {
      float[] vector = new float[dimensions];
      for (int d = 0; d < dimensions; d++) {
        vector[d] = random().nextFloat();
      }
      Document doc = new Document();
      doc.add(new KnnFloatVectorField(FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
      w.addDocument(doc);
    }
  }

  private static void addDocs(IndexWriter w, int count) throws Exception {
    for (int i = 0; i < count; i++) {
      Document doc = new Document();
      doc.add(
          new KnnFloatVectorField(
              FIELD, new float[] {i, i + 1f, i + 2f, i + 3f}, VectorSimilarityFunction.EUCLIDEAN));
      w.addDocument(doc);
    }
  }

  /** The largest number of arcs stored for any node on the given level. */
  private static int widestRowOnLevel(HnswGraph graph, int level) throws Exception {
    int widest = 0;
    HnswGraph.NodesIterator nodes = graph.getNodesOnLevel(level);
    while (nodes.hasNext()) {
      graph.seek(level, nodes.nextInt());
      int arcs = 0;
      while (graph.nextNeighbor() != NO_MORE_DOCS) {
        arcs++;
      }
      widest = Math.max(widest, arcs);
    }
    return widest;
  }

  /** The largest number of arcs stored for any node on any level. */
  private static int widestAdjacencyRow(HnswGraph graph) throws Exception {
    int widest = 0;
    for (int level = 0; level < graph.numLevels(); level++) {
      HnswGraph.NodesIterator nodes = graph.getNodesOnLevel(level);
      while (nodes.hasNext()) {
        int node = nodes.nextInt();
        graph.seek(level, node);
        int arcs = 0;
        while (graph.nextNeighbor() != NO_MORE_DOCS) {
          arcs++;
        }
        widest = Math.max(widest, arcs);
      }
    }
    return widest;
  }

  private static HnswGraph graphOf(LeafReader leaf) throws Exception {
    KnnVectorsReader knnReader = ((CodecReader) leaf).getVectorReader();
    if (knnReader instanceof PerFieldKnnVectorsFormat.FieldsReader fieldsReader) {
      knnReader = fieldsReader.getFieldReader(FIELD);
    }
    return ((HnswGraphProvider) knnReader).getGraph(FIELD);
  }
}

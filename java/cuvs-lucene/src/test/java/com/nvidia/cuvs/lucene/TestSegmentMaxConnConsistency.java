/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
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
import org.apache.lucene.util.hnsw.HnswGraph;
import org.junit.Test;

/**
 * Verifies that the HNSW {@code M} recorded in a segment's metadata describes the graph that
 * segment actually contains, for every segment an accelerated-HNSW writer can produce.
 *
 * <p>{@code M} is written as {@code ceil(cagraGraphDegree / 2)} and bounds what the reader accepts:
 * Lucene sizes its arc buffer as {@code M * 2} and asserts that every stored adjacency row fits. An
 * {@code M} taken from configuration rather than from the built graph can understate the graph,
 * because cuVS is free to build a degree other than the one requested -- and under the HEURISTIC
 * strategy it derives the degree from maxConn and ignores the configured graph degree outright.
 *
 * <p>The invariant checked here is per-segment self-consistency, not cross-segment equality.
 * Segments of the same field legitimately record different values of {@code M}: cuVS truncates the
 * CAGRA graph degree to {@code dataset_size - 1} for small datasets, so with maxConn 16 a 20-vector
 * segment records {@code M = 10} while a 3000-vector segment records {@code M = 16}.
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
   * Every segment -- including a degenerate single-vector one -- must record an M consistent with
   * its own widest adjacency row.
   */
  @Test
  public void testRecordedMMatchesEachSegmentsGraph() throws Exception {
    assumeTrue("cuVS not supported", isSupported());

    // Sizes span the interesting cases: the single-vector special path, a segment small enough for
    // cuVS to truncate the degree, and one large enough to keep the derived degree.
    int[] segmentSizes = {1, 20, 3000};

    try (Directory dir = newDirectory()) {
      IndexWriterConfig cfg =
          new IndexWriterConfig()
              .setCodec(
                  new Lucene101AcceleratedHNSWCodec(
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
                  + " arcs",
              Math.ceilDiv(widestRow, 2),
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
      IndexWriterConfig cfg =
          new IndexWriterConfig().setCodec(new Lucene101AcceleratedHNSWCodec(params));
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        addDocs(w, 1);
        w.commit();
      }

      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        LeafReader leaf = getOnlyLeafReader(reader);
        HnswGraph graph = graphOf(leaf);
        assertEquals(
            "single-vector segment must not record M derived from the ignored graphDegree",
            Math.ceilDiv(widestAdjacencyRow(graph), 2),
            graph.maxConn());
        assertNotEquals("M leaked from the configured graphDegree", 256 / 2, graph.maxConn());
      }
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

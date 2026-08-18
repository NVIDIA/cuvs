/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.generateDataset;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import java.util.Random;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.junit.Test;

/**
 * Repro for the CI-observed {@code EOFException} in {@code OffHeapFloatVectorValues} during
 * concurrent KNN search over an accelerated-HNSW index built via {@link
 * TestAcceleratedHNSWDeletedDocuments}/{@link TestCuVSAcceleratedHNSWDeletedDocuments} (both:
 * deletions + a real merge, heap-buffered path, no {@code numInputVectors}).
 *
 * <p>Rather than relying on a random concurrent search happening to traverse a bad graph node
 * (which only reproduced intermittently, on one CI node), this walks the <em>entire</em> merged
 * HNSW graph directly and asserts every neighbor ordinal is within the merged segment's actual
 * flat-vector count. This targets the suspected root cause: {@code
 * Lucene99AcceleratedHNSWVectorsWriter#mergeOneField} derives the merged vector set <b>twice</b>,
 * independently -- once via the real {@code flatVectorsWriter.mergeOneField} (the authoritative
 * flat {@code .vec} file) and again via {@code vectorBasedMerge}'s own call to {@code
 * KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues} to build the CAGRA/HNSW graph. If
 * those two independently-derived views of "the merged, post-deletion vector set" ever disagree in
 * count or ordinal order, the graph ends up referencing ordinals the flat file doesn't actually
 * have, which is exactly what an out-of-bounds read (EOFException) during traversal would look
 * like.
 *
 * <p>This test is deterministic: it fails on any disagreement between the graph and the flat file,
 * rather than depending on a search happening to reach the bad node.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestMergedGraphOrdinalBounds extends LuceneTestCase {

  private static final String ID_FIELD = "id";
  private static final String FIELD = "vector";

  @Test
  public void testMergedGraphOrdinalsStayWithinFlatVectorBounds() throws Exception {
    assumeTrue("cuVS not supported", isSupported());

    Random random = new Random(1234);
    int segmentSize = 300;
    int dimension = 32;
    // Interspersed deletions on both segments, so the merge must drop a scattered subset of
    // ordinals from each -- not just a contiguous prefix/suffix -- when it re-derives the merged
    // vector set.
    int deleteEveryNth = 4;

    Codec codec = TestUtil.alwaysKnnVectorsFormat(new Lucene99AcceleratedHNSWVectorsFormat());
    IndexWriterConfig config =
        new IndexWriterConfig().setCodec(codec).setMergePolicy(NoMergePolicy.INSTANCE);

    int expectedLiveVectors;
    try (Directory dir = newDirectory();
        IndexWriter writer = new IndexWriter(dir, config)) {
      int deletedFromSegment1 =
          addSegmentWithInterspersedDeletions(
              writer, 0, segmentSize, dimension, deleteEveryNth, random);
      writer.commit(); // segment 1, alone
      int deletedFromSegment2 =
          addSegmentWithInterspersedDeletions(
              writer, segmentSize, segmentSize, dimension, deleteEveryNth, random);
      writer.commit(); // segment 2, alone

      expectedLiveVectors = 2 * segmentSize - deletedFromSegment1 - deletedFromSegment2;

      // NoMergePolicy blocks forced merges too, so swap it out now that the two segments (each
      // with their own interspersed deletions already committed) are set up.
      writer.getConfig().setMergePolicy(new TieredMergePolicy());
      writer.forceMerge(1);
      writer.commit();

      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        assertEquals(
            "expected the forced merge to produce a single segment", 1, reader.leaves().size());
        LeafReader leaf = reader.leaves().get(0).reader();

        FloatVectorValues flatValues = leaf.getFloatVectorValues(FIELD);
        assertEquals(
            "merged flat vector count should equal (added - deleted)",
            expectedLiveVectors,
            flatValues.size());

        HnswGraph graph = graphOf(leaf);
        int level0NodeCount = graph.getNodesOnLevel(0).size();
        assertEquals(
            "HNSW graph's level-0 node count disagrees with the merged flat vector file's actual"
                + " count -- the graph and the flat file were derived independently by"
                + " vectorBasedMerge and flatVectorsWriter.mergeOneField and disagree",
            flatValues.size(),
            level0NodeCount);

        assertAllNeighborOrdinalsInBounds(graph, flatValues.size());
      }
    }
  }

  /**
   * Every neighbor referenced anywhere in the graph, at every level, must be a valid ordinal into
   * the merged segment's actual flat vector data -- otherwise a reader resolving that neighbor's
   * vector (e.g. mid-search, to score it) reads past the end of the flat file.
   */
  private static void assertAllNeighborOrdinalsInBounds(HnswGraph graph, int liveVectorCount)
      throws Exception {
    for (int level = 0; level < graph.numLevels(); level++) {
      HnswGraph.NodesIterator nodes = graph.getNodesOnLevel(level);
      while (nodes.hasNext()) {
        int node = nodes.nextInt();
        assertTrue(
            "node "
                + node
                + " at level "
                + level
                + " is itself out of bounds (live vectors: "
                + liveVectorCount
                + ")",
            node >= 0 && node < liveVectorCount);
        graph.seek(level, node);
        for (int neighbor = graph.nextNeighbor();
            neighbor != NO_MORE_DOCS;
            neighbor = graph.nextNeighbor()) {
          assertTrue(
              "node "
                  + node
                  + " at level "
                  + level
                  + " has a neighbor ordinal "
                  + neighbor
                  + " out of bounds for the merged segment's "
                  + liveVectorCount
                  + " live vectors",
              neighbor >= 0 && neighbor < liveVectorCount);
        }
      }
    }
  }

  private static HnswGraph graphOf(LeafReader leaf) throws Exception {
    KnnVectorsReader knnReader = ((CodecReader) leaf).getVectorReader();
    if (knnReader instanceof PerFieldKnnVectorsFormat.FieldsReader fieldsReader) {
      knnReader = fieldsReader.getFieldReader(FIELD);
    }
    return ((HnswGraphProvider) knnReader).getGraph(FIELD);
  }

  /**
   * Adds {@code count} documents (global ids {@code [startId, startId + count)}), then deletes
   * every {@code deleteEveryNth}-th one by id, scattering the deletions across the segment rather
   * than leaving a contiguous surviving range.
   *
   * @return the number of documents deleted from this segment
   */
  private static int addSegmentWithInterspersedDeletions(
      IndexWriter writer, int startId, int count, int dimension, int deleteEveryNth, Random random)
      throws Exception {
    float[][] dataset = generateDataset(random, count, dimension);
    for (int i = 0; i < count; i++) {
      Document document = new Document();
      document.add(new StringField(ID_FIELD, Integer.toString(startId + i), Field.Store.YES));
      document.add(new KnnFloatVectorField(FIELD, dataset[i], EUCLIDEAN));
      writer.addDocument(document);
    }
    int deleted = 0;
    for (int i = 0; i < count; i += deleteEveryNth) {
      writer.deleteDocuments(new Term(ID_FIELD, Integer.toString(startId + i)));
      deleted++;
    }
    return deleted;
  }
}

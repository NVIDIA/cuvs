/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

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
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.junit.Test;

@SuppressSysoutChecks(bugUrl = "")
public class TestWriterThreadsPersistedIndex extends LuceneTestCase {

  private static final String VECTOR_FIELD = "vector";
  private static final int VECTOR_COUNT = AcceleratedHNSWUtils.PARALLEL_MIN_NODES + 1;
  private static final int DIMENSIONS = 32;

  @Test
  public void testParallelGraphRoundTripAboveThreshold() throws Exception {
    assumeTrue("cuVS not supported", isSupported());
    AcceleratedHNSWParams params =
        new AcceleratedHNSWParams.Builder()
            .withWriterThreads(4)
            .withStrategy(AcceleratedHNSWParams.Strategy.CUSTOM)
            .withIntermediateGraphDegree(32)
            .withGraphDegree(16)
            .withHNSWLayer(1)
            .build();
    Codec codec = new Lucene101AcceleratedHNSWCodec(params);
    float[] query = null;
    Random random = new Random(0x2594L);

    try (Directory directory = newDirectory()) {
      IndexWriterConfig config =
          new IndexWriterConfig()
              .setCodec(codec)
              .setUseCompoundFile(false)
              .setMaxBufferedDocs(VECTOR_COUNT + 1)
              .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      try (IndexWriter writer = new IndexWriter(directory, config)) {
        for (int id = 0; id < VECTOR_COUNT; id++) {
          float[] vector = new float[DIMENSIONS];
          for (int dimension = 0; dimension < DIMENSIONS; dimension++) {
            vector[dimension] = random.nextFloat();
          }
          if (id == 0) {
            query = vector.clone();
          }
          Document document = new Document();
          document.add(new StringField("id", Integer.toString(id), Field.Store.YES));
          document.add(new KnnFloatVectorField(VECTOR_FIELD, vector, EUCLIDEAN));
          writer.addDocument(document);
        }
      }

      TestUtil.checkIndex(directory);
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        assertEquals(1, reader.leaves().size());
        assertEquals(VECTOR_COUNT, reader.numDocs());
        HnswGraph graph = graphOf(getOnlyLeafReader(reader));
        assertEquals(VECTOR_COUNT, graph.size());
        int arcs = 0;
        HnswGraph.NodesIterator nodes = graph.getNodesOnLevel(0);
        while (nodes.hasNext()) {
          int node = nodes.nextInt();
          graph.seek(0, node);
          for (int neighbor = graph.nextNeighbor();
              neighbor != NO_MORE_DOCS;
              neighbor = graph.nextNeighbor()) {
            assertTrue(neighbor >= 0);
            assertTrue(neighbor < VECTOR_COUNT);
            arcs++;
          }
        }
        assertTrue("persisted graph contains no arcs", arcs > 0);

        IndexSearcher searcher = new IndexSearcher(reader);
        var hits = searcher.search(new KnnFloatVectorQuery(VECTOR_FIELD, query, 10), 10);
        assertEquals(10, hits.scoreDocs.length);
        boolean foundExactVector = false;
        for (var hit : hits.scoreDocs) {
          foundExactVector |= "0".equals(searcher.storedFields().document(hit.doc).get("id"));
        }
        assertTrue("the indexed vector must be returned for its own query", foundExactVector);
      }
    }
  }

  private static HnswGraph graphOf(LeafReader leaf) throws Exception {
    KnnVectorsReader reader = ((CodecReader) leaf).getVectorReader();
    if (reader instanceof PerFieldKnnVectorsFormat.FieldsReader fieldsReader) {
      reader = fieldsReader.getFieldReader(VECTOR_FIELD);
    }
    return ((HnswGraphProvider) reader).getGraph(VECTOR_FIELD);
  }
}

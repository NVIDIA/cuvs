/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import com.carrotsearch.randomizedtesting.annotations.Name;
import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
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
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.junit.Test;

@SuppressSysoutChecks(bugUrl = "")
public class TestGraphThreadsPersistedIndex extends LuceneTestCase {

  private static final String VECTOR_FIELD = "vector";
  private static final String REQUIRE_GPU_ENV = "CUVS_TESTS_REQUIRE_GPU";
  private static final int VECTOR_COUNT = GPUBuiltHnswGraph.PARALLEL_MIN_NODES + 1;
  private static final int GRAPH_THREADS = 4;

  private enum WriterVariant {
    FLOAT,
    BINARY_QUANTIZED,
    SCALAR_QUANTIZED
  }

  private final WriterVariant writerVariant;
  private final int dimensions;

  public TestGraphThreadsPersistedIndex(
      @Name("writer") WriterVariant writerVariant, @Name("dimensions") int dimensions) {
    this.writerVariant = writerVariant;
    this.dimensions = dimensions;
  }

  @ParametersFactory
  public static List<Object[]> parameters() {
    return List.of(
        new Object[] {WriterVariant.FLOAT, 32},
        new Object[] {WriterVariant.BINARY_QUANTIZED, 129},
        new Object[] {WriterVariant.SCALAR_QUANTIZED, 32});
  }

  @Test
  public void testEveryWriterForwardsGraphThreadsAndPersistsValidIndex() throws Exception {
    requireCuvsSupport();
    AcceleratedHNSWParams params =
        new AcceleratedHNSWParams.Builder()
            .withWriterThreads(1)
            .withGraphThreads(GRAPH_THREADS)
            .withStrategy(AcceleratedHNSWParams.Strategy.CUSTOM)
            .withIntermediateGraphDegree(32)
            .withGraphDegree(16)
            .withHNSWLayer(1)
            .build();
    Codec codec = codecFor(params);
    RecordingInfoStream infoStream = new RecordingInfoStream();

    try (Directory directory = newDirectory()) {
      writeIndexWithGraphThreads(directory, codec, infoStream);
      assertParallelStageObserved(infoStream, "materialization", "device-host-copy");
      assertParallelStageObserved(infoStream, "serialization", "above-threshold");
      assertPersistedGraphIsValidAndSearchable(directory);
    }
  }

  private void writeIndexWithGraphThreads(
      Directory directory, Codec codec, RecordingInfoStream infoStream) throws Exception {
    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(codec)
            .setInfoStream(infoStream)
            .setUseCompoundFile(false)
            .setMaxBufferedDocs(VECTOR_COUNT + 1)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);
    Random random = new Random(0x2594L);
    try (IndexWriter writer = new IndexWriter(directory, config)) {
      for (int id = 0; id < VECTOR_COUNT; id++) {
        float[] vector = new float[dimensions];
        for (int dimension = 0; dimension < dimensions; dimension++) {
          vector[dimension] = random.nextFloat();
        }
        Document document = new Document();
        document.add(new StringField("id", Integer.toString(id), Field.Store.YES));
        document.add(new KnnFloatVectorField(VECTOR_FIELD, vector, EUCLIDEAN));
        writer.addDocument(document);
      }
    }
  }

  private static void assertPersistedGraphIsValidAndSearchable(Directory directory)
      throws Exception {
    TestUtil.checkIndex(directory);
    try (DirectoryReader reader = DirectoryReader.open(directory)) {
      assertEquals(1, reader.leaves().size());
      assertEquals(VECTOR_COUNT, reader.numDocs());
      LeafReader leaf = getOnlyLeafReader(reader);
      HnswGraph graph = graphOf(leaf);
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

      int queryNode = graph.entryNode();
      assertTrue(queryNode >= 0);
      assertTrue(queryNode < VECTOR_COUNT);
      FloatVectorValues values = leaf.getFloatVectorValues(VECTOR_FIELD);
      assertNotNull(values);
      float[] query = values.vectorValue(queryNode).clone();
      int queryDoc = values.ordToDoc(queryNode);
      String queryId = leaf.storedFields().document(queryDoc).get("id");

      IndexSearcher searcher = new IndexSearcher(reader);
      var hits = searcher.search(new KnnFloatVectorQuery(VECTOR_FIELD, query, 10), 10);
      assertEquals(10, hits.scoreDocs.length);
      boolean foundQueryNode = false;
      for (var hit : hits.scoreDocs) {
        foundQueryNode |= queryId.equals(searcher.storedFields().document(hit.doc).get("id"));
      }
      assertTrue("the entry-node vector must be returned for its own query", foundQueryNode);
    }
  }

  private static void requireCuvsSupport() {
    boolean supported = isSupported();
    if ("1".equals(System.getenv(REQUIRE_GPU_ENV))) {
      assertTrue(
          REQUIRE_GPU_ENV
              + "=1, but cuVS failed to initialize; verify GPU visibility, matching "
              + "libcuvs/libcuvs_c libraries, LD_LIBRARY_PATH, and Java native-access "
              + "configuration.",
          supported);
    } else {
      assumeTrue("cuVS not supported", supported);
    }
  }

  private Codec codecFor(AcceleratedHNSWParams params) throws Exception {
    return switch (writerVariant) {
      case FLOAT -> new Lucene101AcceleratedHNSWCodec(params);
      case BINARY_QUANTIZED -> new LuceneAcceleratedHNSWBinaryQuantizedCodec(params);
      case SCALAR_QUANTIZED -> new LuceneAcceleratedHNSWScalarQuantizedCodec(params);
    };
  }

  private void assertParallelStageObserved(
      RecordingInfoStream infoStream, String stage, String expectedReason) {
    String expectedPath =
        "graph-processing stage="
            + stage
            + " mode=parallel reason="
            + expectedReason
            + " requestedThreads="
            + GRAPH_THREADS
            + " nodes="
            + VECTOR_COUNT;
    assertTrue(
        writerVariant + " did not report the expected path; messages: " + infoStream.messages(),
        infoStream.messages().stream().anyMatch(message -> message.contains(expectedPath)));
  }

  private static HnswGraph graphOf(LeafReader leaf) throws Exception {
    KnnVectorsReader reader = ((CodecReader) leaf).getVectorReader();
    if (reader instanceof PerFieldKnnVectorsFormat.FieldsReader fieldsReader) {
      reader = fieldsReader.getFieldReader(VECTOR_FIELD);
    }
    return ((HnswGraphProvider) reader).getGraph(VECTOR_FIELD);
  }

  private static final class RecordingInfoStream extends InfoStream {

    private final List<String> messages = Collections.synchronizedList(new ArrayList<>());

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

    List<String> messages() {
      synchronized (messages) {
        return List.copyOf(messages);
      }
    }
  }
}

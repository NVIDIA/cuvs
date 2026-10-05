/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.generateDataset;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;
import java.util.UUID;
import org.apache.commons.io.FileUtils;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/**
 * Regression test for the CAGRA-to-HNSW conversion when the CAGRA graph has an <b>odd</b> graph
 * degree.
 *
 * <p>The HNSW max-connections parameter is derived from the CAGRA graph degree as {@code M =
 * ceil(degree / 2)}, and Lucene's HNSW format allows up to {@code 2 * M} neighbors at level 0. With
 * a plain integer {@code degree / 2}, an odd degree {@code d} yields {@code 2 * (d / 2) = d - 1 < d}
 * — one fewer than the number of neighbors each node actually has — which makes the writer emit a
 * graph the reader rejects with "too many neighbors: d". cuVS produces odd graph degrees when it
 * clamps the degree for small datasets; here we force one deterministically via the CUSTOM strategy.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestAcceleratedHNSWOddGraphDegree extends LuceneTestCase {

  private static final String VECTOR_FIELD = "vector_field";
  private static final int ODD_GRAPH_DEGREE = 63;

  private Random random;
  private Path indexDirPath;

  @Before
  public void beforeTest() {
    assumeTrue("cuVS not supported", isSupported());
    random = new Random(222);
    indexDirPath = Paths.get(UUID.randomUUID().toString());
  }

  @Test
  public void testOddGraphDegreeIndexesAndSearches() throws Exception {
    // CUSTOM strategy passes the graph degree straight through to cuVS, so the built CAGRA graph
    // (and thus its adjacency list's column count) has an odd degree.
    AcceleratedHNSWParams params =
        new AcceleratedHNSWParams.Builder()
            .withStrategy(AcceleratedHNSWParams.Strategy.CUSTOM)
            .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
            .withIntermediateGraphDegree(128)
            .withGraphDegree(ODD_GRAPH_DEGREE)
            .build();

    Codec codec = new Lucene101AcceleratedHNSWCodec(params);
    IndexWriterConfig config = new IndexWriterConfig().setCodec(codec).setUseCompoundFile(false);

    int numDocs = 1000;
    int dimension = 32;
    int topK = 10;
    float[][] dataset = generateDataset(random, numDocs, dimension);

    try (Directory indexDirectory = FSDirectory.open(indexDirPath)) {
      // Indexing (flush of the HNSW graph is where an odd degree would trip "too many neighbors").
      try (IndexWriter indexWriter = new IndexWriter(indexDirectory, config)) {
        for (int i = 0; i < numDocs; i++) {
          Document document = new Document();
          document.add(new KnnFloatVectorField(VECTOR_FIELD, dataset[i], EUCLIDEAN));
          indexWriter.addDocument(document);
        }
        indexWriter.commit();
      }

      // Searching (the Lucene HNSW reader validates neighbor counts against the stored M).
      try (DirectoryReader reader = DirectoryReader.open(indexDirectory)) {
        assertEquals(numDocs, reader.numDocs());
        IndexSearcher searcher = new IndexSearcher(reader);
        float[] queryVector = generateDataset(random, 1, dimension)[0];
        TopDocs results =
            searcher.search(new KnnFloatVectorQuery(VECTOR_FIELD, queryVector, topK), topK);
        assertEquals(topK, results.scoreDocs.length);
      }
    }
  }

  @After
  public void afterTest() throws Exception {
    // JUnit runs @After even when @Before ends in a skipped assumption, at which point the path was
    // never assigned. Dereferencing it would turn the skip into a failure on machines without cuVS.
    if (indexDirPath == null) {
      return;
    }
    var dir = indexDirPath.toFile();
    if (dir.exists() && dir.isDirectory()) {
      FileUtils.deleteDirectory(dir);
    }
  }
}

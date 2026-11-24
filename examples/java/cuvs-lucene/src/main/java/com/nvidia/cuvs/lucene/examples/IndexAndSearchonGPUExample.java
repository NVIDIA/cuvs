/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// fern/pages/user_guide/lucene.md embeds this file in the docs from the package line on, which it
// finds by line number. Adding or removing lines above it requires updating `lines` there.
package com.nvidia.cuvs.lucene.examples;

import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import com.nvidia.cuvs.lucene.CuVSCodecs;
import com.nvidia.cuvs.lucene.GPUKnnFloatVectorQuery;
import com.nvidia.cuvs.lucene.GPUSearchParams;
import com.nvidia.cuvs.spi.CuVSProvider;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

public class IndexAndSearchonGPUExample {

  private static final String ID_FIELD = "id";
  private static final String VECTOR_FIELD = "vector_field";
  private static final String STATUS_FIELD = "status";

  public static void main(String[] args) throws Exception {
    // Select the process-wide RMM allocator before creating any cuVS resources or codecs.
    CuVSProvider.provider().enableRMMAsyncMemory();

    int numDocs = 2000;
    int dimension = 32;
    int topK = 5;

    Random random = new Random(222);
    float[][] dataset = generateDataset(random, numDocs, dimension);
    Path indexDirPath = Files.createTempDirectory("cuvs-lucene-gpu-search");

    GPUSearchParams params = new GPUSearchParams.Builder().build();
    Codec codec = CuVSCodecs.gpuSearch(params);
    IndexWriterConfig config = new IndexWriterConfig().setCodec(codec);

    // Indexing
    try (Directory indexDirectory = FSDirectory.open(indexDirPath);
        IndexWriter indexWriter = new IndexWriter(indexDirectory, config)) {
      for (int i = 0; i < numDocs; i++) {
        Document document = new Document();
        document.add(new StringField(ID_FIELD, Integer.toString(i), Field.Store.YES));
        document.add(new KnnFloatVectorField(VECTOR_FIELD, dataset[i], EUCLIDEAN));
        document.add(
            new StringField(STATUS_FIELD, i % 2 == 0 ? "active" : "in-active", Field.Store.YES));
        indexWriter.addDocument(document);
      }
    }

    // Searching, restricted to the documents matching the filter
    try (Directory indexDirectory = FSDirectory.open(indexDirPath);
        DirectoryReader reader = DirectoryReader.open(indexDirectory)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      float[] queryVector = generateDataset(random, 1, dimension)[0];
      Query filter = new TermQuery(new Term(STATUS_FIELD, "active"));
      KnnFloatVectorQuery query =
          new GPUKnnFloatVectorQuery(VECTOR_FIELD, queryVector, topK, filter, topK, 1);
      TopDocs results = searcher.search(query, topK);

      for (ScoreDoc scoreDoc : results.scoreDocs) {
        Document document = searcher.storedFields().document(scoreDoc.doc);
        System.out.println(
            "doc "
                + document.get(ID_FIELD)
                + " status="
                + document.get(STATUS_FIELD)
                + " score="
                + scoreDoc.score);
      }
    }
  }

  private static float[][] generateDataset(Random random, int size, int dimensions) {
    float[][] dataset = new float[size][dimensions];
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < dimensions; j++) {
        dataset[i][j] = random.nextFloat() * 100;
      }
    }
    return dataset;
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// fern/pages/user_guide/lucene.md embeds this file in the docs from the package line on, which it
// finds by line number. Adding or removing lines above it requires updating `lines` there.
package com.nvidia.cuvs.lucene.examples;

import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import com.nvidia.cuvs.lucene.AcceleratedHNSWParams;
import com.nvidia.cuvs.lucene.CuVSCodecs;
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
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

public class AcceleratedHnswExample {

  private static final String ID_FIELD = "id";
  private static final String VECTOR_FIELD = "vector_field";

  public static void main(String[] args) throws Exception {
    int numDocs = 2000;
    int dimension = 32;
    int topK = 5;

    Random random = new Random(222);
    float[][] dataset = generateDataset(random, numDocs, dimension);
    Path indexDirPath = Files.createTempDirectory("cuvs-lucene-quickstart");

    AcceleratedHNSWParams params = new AcceleratedHNSWParams.Builder().build();
    Codec codec = CuVSCodecs.acceleratedHNSW(params);
    IndexWriterConfig config = new IndexWriterConfig().setCodec(codec);

    // Indexing
    try (Directory indexDirectory = FSDirectory.open(indexDirPath);
        IndexWriter indexWriter = new IndexWriter(indexDirectory, config)) {
      for (int i = 0; i < numDocs; i++) {
        Document document = new Document();
        document.add(new StringField(ID_FIELD, Integer.toString(i), Field.Store.YES));
        document.add(new KnnFloatVectorField(VECTOR_FIELD, dataset[i], EUCLIDEAN));
        indexWriter.addDocument(document);
      }
    }

    // Searching
    try (Directory indexDirectory = FSDirectory.open(indexDirPath);
        DirectoryReader reader = DirectoryReader.open(indexDirectory)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      float[] queryVector = generateDataset(random, 1, dimension)[0];
      KnnFloatVectorQuery query = new KnnFloatVectorQuery(VECTOR_FIELD, queryVector, topK);
      TopDocs results = searcher.search(query, topK);

      for (ScoreDoc scoreDoc : results.scoreDocs) {
        Document document = searcher.storedFields().document(scoreDoc.doc);
        System.out.println("doc " + document.get(ID_FIELD) + " score=" + scoreDoc.score);
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

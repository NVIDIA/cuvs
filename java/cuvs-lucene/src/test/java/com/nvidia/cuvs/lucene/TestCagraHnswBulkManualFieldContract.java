/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.apache.lucene.index.VectorSimilarityFunction.COSINE;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

/** Fail-closed field-contract tests for the public manual bulk writer API. */
public class TestCagraHnswBulkManualFieldContract extends LuceneTestCase {

  private static final String VECTOR_FIELD = "vector";
  private static final int DIMENSIONS = 8;

  @Test
  public void testRejectsWrongVectorFieldName() throws Exception {
    Document document = vectorDocument("unexpected", DIMENSIONS, EUCLIDEAN);

    assertRejected(document, "field name");
  }

  @Test
  public void testRejectsWrongVectorDimension() throws Exception {
    Document document = vectorDocument(VECTOR_FIELD, DIMENSIONS - 1, EUCLIDEAN);

    assertRejected(document, "dimension");
  }

  @Test
  public void testRejectsWrongVectorSimilarity() throws Exception {
    Document document = vectorDocument(VECTOR_FIELD, DIMENSIONS, COSINE);

    assertRejected(document, "similarity");
  }

  @Test
  public void testRejectsDocumentWithoutVector() throws Exception {
    Document document = new Document();
    document.add(new StringField("id", "0", Field.Store.YES));

    assertRejected(document, "exactly one vector field");
  }

  @Test
  public void testRejectsDocumentWithMoreThanOneVector() throws Exception {
    Document document = vectorDocument(VECTOR_FIELD, DIMENSIONS, EUCLIDEAN);
    document.add(new KnnFloatVectorField(VECTOR_FIELD, new float[DIMENSIONS], EUCLIDEAN));

    assertRejected(document, "exactly one vector field");
  }

  private static Document vectorDocument(
      String fieldName,
      int dimensions,
      org.apache.lucene.index.VectorSimilarityFunction similarity) {
    Document document = new Document();
    document.add(new KnnFloatVectorField(fieldName, new float[dimensions], similarity));
    return document;
  }

  private static void assertRejected(Document document, String expectedMessage) throws Exception {
    try (Directory directory = newDirectory();
        CagraHnswBulkIndexWriter writer =
            new CagraHnswBulkIndexWriter(directory, new IndexWriterConfig(), config(), 1)) {
      IllegalArgumentException failure;
      try {
        failure = expectThrows(IllegalArgumentException.class, () -> writer.addDocument(document));
      } finally {
        writer.abort();
      }
      assertTrue(failure.getMessage(), failure.getMessage().contains(expectedMessage));
    }
  }

  private static CagraHnswBulkIndexWriter.Config config() {
    return CagraHnswBulkIndexWriter.Config.builder()
        .field(VECTOR_FIELD, DIMENSIONS, EUCLIDEAN)
        .graphBuild(new AcceleratedHNSWParams.Builder().build())
        .build();
  }
}

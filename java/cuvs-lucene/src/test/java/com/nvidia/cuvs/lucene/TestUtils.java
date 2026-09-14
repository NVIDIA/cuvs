/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.StoredFields;

public class TestUtils {

  /**
   * Asserts that every vector in {@code field} is still paired with the document that indexed it,
   * matching on the document's stored {@code id} rather than on ordinal position.
   *
   * <p>Nothing fixes the order documents land in after a merge, and the randomized test
   * framework's {@code MockRandomMergePolicy} disturbs it two independent ways: {@code
   * findForcedMerges} shuffles the segments it is about to merge, and {@code
   * MockRandomOneMerge.reorder} reverses doc IDs outright. The second applies even when there is
   * only one segment, so a test that never commits between documents is no safer than one that
   * does. Asserting {@code vectorValue(i) == expectedById[i]} therefore fails on some seeds with
   * nothing wrong. What has to hold is that no document loses its own vector, which is what this
   * checks.
   *
   * @param expectedById the vector indexed for each document id, indexed by that id
   */
  public static void assertVectorsKeepTheirDocuments(
      LeafReader reader, String field, float[][] expectedById) throws IOException {
    FloatVectorValues values = reader.getFloatVectorValues(field);
    assertNotNull("no vector values for field " + field, values);
    assertEquals(expectedById.length, values.size());
    StoredFields storedFields = reader.storedFields();
    Set<Integer> seen = new HashSet<>();
    int previousDoc = -1;
    for (int ord = 0; ord < values.size(); ord++) {
      int doc = values.ordToDoc(ord);
      // Ordinals are assigned in docID order. Where every document has a vector -- every caller
      // today -- ordToDoc is the hardcoded identity and this cannot fire. It earns its keep only
      // if a caller passes a field that some documents lack, where the id lookup below would
      // otherwise be free to agree with a garbled mapping.
      assertTrue("ordToDoc went backwards at ordinal " + ord, doc > previousDoc);
      previousDoc = doc;
      String storedId = storedFields.document(doc).get("id");
      assertNotNull("document at ordinal " + ord + " has no stored id", storedId);
      int id = Integer.parseInt(storedId);
      assertTrue("document id " + id + " appeared twice", seen.add(id));
      assertArrayEquals(
          "vector for document id " + id + " (doc " + doc + ", ordinal " + ord + ")",
          expectedById[id],
          values.vectorValue(ord),
          0.0f);
    }
    assertEquals("not every document was found", expectedById.length, seen.size());
  }

  public static float[][] generateDataset(Random random, int size, int dimensions) {
    float[][] dataset = new float[size][dimensions];
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < dimensions; j++) {
        dataset[i][j] = random.nextFloat() * 100;
      }
    }
    return dataset;
  }

  public static float[] generateRandomVector(int dimensions, Random random) {
    float[] vector = new float[dimensions];
    for (int i = 0; i < dimensions; i++) {
      vector[i] = random.nextFloat() * 100;
    }
    return vector;
  }

  public static float[][] generateQueries(Random random, int dimensions, int numQueries) {
    // Generate random query vectors
    float[][] queries = new float[numQueries][dimensions];
    for (int i = 0; i < numQueries; i++) {
      for (int j = 0; j < dimensions; j++) {
        queries[i][j] = random.nextFloat() * 100;
      }
    }
    return queries;
  }
}

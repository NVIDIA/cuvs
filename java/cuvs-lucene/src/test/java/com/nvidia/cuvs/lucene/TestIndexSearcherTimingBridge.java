/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.junit.Test;

public class TestIndexSearcherTimingBridge {

  @Test
  public void testReturnsTopDocsAndInJvmElapsedTimeThroughStandardTypes() throws Exception {
    try (var directory = new ByteBuffersDirectory();
        var writer = new IndexWriter(directory, new IndexWriterConfig())) {
      writer.addDocument(new Document());
      writer.commit();

      try (var reader = DirectoryReader.open(directory)) {
        Function<Map<String, Object>, Map<String, Object>> bridge = new IndexSearcherTimingBridge();
        Map<String, Object> request = new HashMap<>();
        request.put(IndexSearcherTimingBridge.SEARCHER_KEY, new IndexSearcher(reader));
        request.put(IndexSearcherTimingBridge.QUERY_KEY, new MatchAllDocsQuery());
        request.put(IndexSearcherTimingBridge.TOP_K_KEY, 1);

        Map<String, Object> response = bridge.apply(request);

        TopDocs topDocs = (TopDocs) response.get(IndexSearcherTimingBridge.TOP_DOCS_KEY);
        Long elapsedNanos = (Long) response.get(IndexSearcherTimingBridge.ELAPSED_NANOS_KEY);
        assertNotNull(topDocs);
        assertEquals(1, topDocs.scoreDocs.length);
        assertNotNull(elapsedNanos);
        assertTrue(elapsedNanos >= 0L);
      }
    }
  }

  @Test
  public void testRejectsAnIncompleteRequestBeforeSearching() {
    var request = new HashMap<String, Object>();
    request.put(IndexSearcherTimingBridge.QUERY_KEY, new MatchAllDocsQuery());
    request.put(IndexSearcherTimingBridge.TOP_K_KEY, 1);

    IllegalArgumentException error =
        assertThrows(
            IllegalArgumentException.class, () -> new IndexSearcherTimingBridge().apply(request));

    assertEquals("searcher must have type IndexSearcher", error.getMessage());
  }
}

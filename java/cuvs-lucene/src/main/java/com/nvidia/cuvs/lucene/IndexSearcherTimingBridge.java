/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;

/**
 * Measures one {@link IndexSearcher#search(Query, int)} invocation inside the JVM.
 *
 * <p>The standard {@link Function} and {@link Map} types form a narrow bridge for generated Java
 * bindings that do not wrap this class directly. Each call still represents one ordinary Lucene
 * query; this class does not add batching or concurrency.
 *
 * <p>The request map must contain {@code searcher} ({@link IndexSearcher}), {@code query} ({@link
 * Query}), and {@code top_k} ({@link Integer}). The response contains {@code top_docs} ({@link
 * TopDocs}) and {@code elapsed_nanos} ({@link Long}). An {@link IOException} from Lucene is exposed
 * as an {@link UncheckedIOException} because {@link Function#apply(Object)} cannot declare checked
 * exceptions.
 */
public final class IndexSearcherTimingBridge
    implements Function<Map<String, Object>, Map<String, Object>> {
  public static final String SEARCHER_KEY = "searcher";
  public static final String QUERY_KEY = "query";
  public static final String TOP_K_KEY = "top_k";
  public static final String TOP_DOCS_KEY = "top_docs";
  public static final String ELAPSED_NANOS_KEY = "elapsed_nanos";

  @Override
  public Map<String, Object> apply(Map<String, Object> request) {
    IndexSearcher searcher = requiredValue(request, SEARCHER_KEY, IndexSearcher.class);
    Query query = requiredValue(request, QUERY_KEY, Query.class);
    Integer topK = requiredValue(request, TOP_K_KEY, Integer.class);
    if (topK < 1) {
      throw new IllegalArgumentException("top_k must be positive");
    }

    long started = System.nanoTime();
    TopDocs topDocs;
    try {
      topDocs = searcher.search(query, topK);
    } catch (IOException error) {
      throw new UncheckedIOException("IndexSearcher.search failed", error);
    }
    long elapsedNanos = System.nanoTime() - started;

    Map<String, Object> response = new HashMap<>();
    response.put(TOP_DOCS_KEY, topDocs);
    response.put(ELAPSED_NANOS_KEY, elapsedNanos);
    return response;
  }

  private static <T> T requiredValue(
      Map<String, Object> values, String key, Class<T> expectedType) {
    if (values == null) {
      throw new IllegalArgumentException("request must not be null");
    }
    Object value = values.get(key);
    if (!expectedType.isInstance(value)) {
      throw new IllegalArgumentException(key + " must have type " + expectedType.getSimpleName());
    }
    return expectedType.cast(value);
  }
}

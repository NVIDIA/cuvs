/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.lang.reflect.Field;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;

/**
 * Subprocess entry point for assertions that require fresh process-local class state.
 *
 * <p>Constructor probes inspect, rather than clear, the provider cache so eager initialization
 * cannot be hidden.
 */
public final class SPIColdStartProbe {

  static final String CODEC_SPI_DISCOVERY_MODE = "codec";
  static final String VECTOR_FORMAT_SPI_DISCOVERY_MODE = "knn";
  static final String SCALAR_CONSTRUCTOR_MODE = "scalar-constructor";
  static final String BINARY_CONSTRUCTOR_MODE = "binary-constructor";

  private static final Set<String> CODEC_NAMES =
      Set.of(
          "Lucene101AcceleratedHNSWCodec",
          "CuVS2510GPUSearchCodec",
          "Lucene101AcceleratedHNSWBinaryQuantizedCodec",
          "Lucene101AcceleratedHNSWScalarQuantizedCodec");

  private static final Set<String> VECTOR_FORMAT_NAMES =
      Set.of(
          "CuVS2510GPUVectorsFormat",
          "Lucene99AcceleratedHNSWVectorsFormat",
          "Lucene99AcceleratedHNSWBinaryQuantizedVectorsFormat",
          "Lucene99AcceleratedHNSWScalarQuantizedVectorsFormat");

  private SPIColdStartProbe() {}

  public static void main(String[] args) {
    if (args.length != 1) {
      throw new IllegalArgumentException("Expected one probe mode");
    }
    switch (args[0]) {
      case CODEC_SPI_DISCOVERY_MODE -> assertExpectedCuvsCodecsResolve();
      case VECTOR_FORMAT_SPI_DISCOVERY_MODE -> assertExpectedCuvsVectorFormatsResolve();
      case SCALAR_CONSTRUCTOR_MODE -> assertScalarConstructorLeavesProviderCacheEmpty();
      case BINARY_CONSTRUCTOR_MODE -> assertBinaryConstructorLeavesProviderCacheEmpty();
      default -> throw new IllegalArgumentException("Unknown probe mode: " + args[0]);
    }
  }

  private static void assertExpectedCuvsCodecsResolve() {
    Set<String> available = Codec.availableCodecs();
    assertAllAvailable("codecs", available, CODEC_NAMES);
    for (String name : CODEC_NAMES) {
      assertResolvedNameEquals(name, Codec.forName(name).getName());
    }
  }

  private static void assertExpectedCuvsVectorFormatsResolve() {
    Set<String> available = KnnVectorsFormat.availableKnnVectorsFormats();
    assertAllAvailable("vector formats", available, VECTOR_FORMAT_NAMES);
    for (String name : VECTOR_FORMAT_NAMES) {
      assertResolvedNameEquals(name, KnnVectorsFormat.forName(name).getName());
    }
  }

  private static void assertScalarConstructorLeavesProviderCacheEmpty() {
    new LuceneAcceleratedHNSWScalarQuantizedVectorsFormat();
    assertProviderCacheIsEmptyAfterConstruction("Scalar");
  }

  private static void assertBinaryConstructorLeavesProviderCacheEmpty() {
    new LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat();
    assertProviderCacheIsEmptyAfterConstruction("Binary");
  }

  private static void assertProviderCacheIsEmptyAfterConstruction(String formatName) {
    try {
      Field instancesField = LuceneProvider.class.getDeclaredField("INSTANCES");
      instancesField.setAccessible(true);
      @SuppressWarnings("unchecked")
      Map<String, LuceneProvider> instances =
          (Map<String, LuceneProvider>) instancesField.get(null);
      if (!instances.isEmpty()) {
        throw new AssertionError(
            formatName
                + " format construction initialized Lucene providers: "
                + instances.keySet());
      }
    } catch (ReflectiveOperationException e) {
      throw new AssertionError("Unable to inspect Lucene provider cache", e);
    }
  }

  private static void assertAllAvailable(String kind, Set<String> available, Set<String> expected) {
    if (!available.containsAll(expected)) {
      throw new AssertionError(
          "Missing "
              + kind
              + ": "
              + findMissingNames(expected, available)
              + "; available="
              + available);
    }
  }

  private static Set<String> findMissingNames(Set<String> expected, Set<String> available) {
    HashSet<String> missing = new HashSet<>(expected);
    missing.removeAll(available);
    return missing;
  }

  private static void assertResolvedNameEquals(String expected, String actual) {
    if (!expected.equals(actual)) {
      throw new AssertionError("Expected " + expected + " but resolved " + actual);
    }
  }
}

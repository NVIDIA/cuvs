/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.util.Set;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;

/** Exercises Lucene SPI discovery before another test can initialize its static holders. */
public final class SPIColdStartProbe {

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
      case "codec" -> probeCodecs();
      case "knn" -> probeVectorFormats();
      case "scalar-constructor" -> probeScalarConstructor();
      default -> throw new IllegalArgumentException("Unknown probe mode: " + args[0]);
    }
  }

  private static void probeCodecs() {
    Set<String> available = Codec.availableCodecs();
    requireAll("codecs", available, CODEC_NAMES);
    for (String name : CODEC_NAMES) {
      requireName(name, Codec.forName(name).getName());
    }
  }

  private static void probeVectorFormats() {
    Set<String> available = KnnVectorsFormat.availableKnnVectorsFormats();
    requireAll("vector formats", available, VECTOR_FORMAT_NAMES);
    for (String name : VECTOR_FORMAT_NAMES) {
      requireName(name, KnnVectorsFormat.forName(name).getName());
    }
  }

  private static void probeScalarConstructor() {
    new LuceneAcceleratedHNSWScalarQuantizedVectorsFormat();
    try {
      java.lang.reflect.Field instancesField = LuceneProvider.class.getDeclaredField("INSTANCES");
      instancesField.setAccessible(true);
      @SuppressWarnings("unchecked")
      java.util.Map<String, LuceneProvider> instances =
          (java.util.Map<String, LuceneProvider>) instancesField.get(null);
      if (!instances.isEmpty()) {
        throw new AssertionError(
            "Scalar format construction initialized Lucene providers: " + instances.keySet());
      }
    } catch (ReflectiveOperationException e) {
      throw new AssertionError("Unable to inspect Lucene provider cache", e);
    }
  }

  private static void requireAll(String kind, Set<String> available, Set<String> expected) {
    if (!available.containsAll(expected)) {
      throw new AssertionError(
          "Missing " + kind + ": " + difference(expected, available) + "; available=" + available);
    }
  }

  private static Set<String> difference(Set<String> expected, Set<String> available) {
    java.util.HashSet<String> missing = new java.util.HashSet<>(expected);
    missing.removeAll(available);
    return missing;
  }

  private static void requireName(String expected, String actual) {
    if (!expected.equals(actual)) {
      throw new AssertionError("Expected " + expected + " but resolved " + actual);
    }
  }
}

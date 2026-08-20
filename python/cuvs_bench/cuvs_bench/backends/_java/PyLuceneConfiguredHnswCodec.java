/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.bench;

import com.nvidia.cuvs.CagraIndexParams.HnswHeuristicType;
import com.nvidia.cuvs.lucene.AcceleratedHNSWParams;
import com.nvidia.cuvs.lucene.Lucene101AcceleratedHNSWCodec;

/**
 * PyLucene-compatible adapter for configuring the accelerated HNSW codec.
 *
 * <p>PyLucene instantiates codecs through a no-argument constructor. This adapter reads the two
 * Lucene-equivalent HNSW build parameters from system properties and delegates all codec behavior
 * to the production cuVS-Lucene implementation.
 */
public final class PyLuceneConfiguredHnswCodec extends Lucene101AcceleratedHNSWCodec {

  public static final String M_PROPERTY = "com.nvidia.cuvs.bench.pylucene.hnsw.m";
  public static final String EF_CONSTRUCTION_PROPERTY =
      "com.nvidia.cuvs.bench.pylucene.hnsw.efConstruction";

  private final int m;
  private final int efConstruction;

  /** Constructs the production codec with parameters supplied through system properties. */
  public PyLuceneConfiguredHnswCodec() throws Exception {
    this(configuredValues());
  }

  private PyLuceneConfiguredHnswCodec(ConfiguredValues values) throws Exception {
    super(values.parameters());
    this.m = values.m();
    this.efConstruction = values.efConstruction();
  }

  private static ConfiguredValues configuredValues() {
    int m = requiredIntegerProperty(M_PROPERTY);
    int efConstruction = requiredIntegerProperty(EF_CONSTRUCTION_PROPERTY);
    AcceleratedHNSWParams parameters =
        new AcceleratedHNSWParams.Builder()
            .withStrategy(AcceleratedHNSWParams.Strategy.HEURISTIC)
            .withHnswHeuristicType(HnswHeuristicType.SAME_GRAPH_FOOTPRINT)
            .withMaxConn(m)
            .withBeamWidth(efConstruction)
            .build();
    return new ConfiguredValues(m, efConstruction, parameters);
  }

  private static int requiredIntegerProperty(String name) {
    String value = System.getProperty(name);
    if (value == null) {
      throw new IllegalStateException("Required system property is not set: " + name);
    }

    try {
      return Integer.parseInt(value);
    } catch (NumberFormatException error) {
      throw new IllegalArgumentException(
          "System property " + name + " must be an integer, found: " + value, error);
    }
  }

  @Override
  public String toString() {
    return getClass().getSimpleName() + "(m=" + m + ", efConstruction=" + efConstruction + ")";
  }

  private record ConfiguredValues(int m, int efConstruction, AcceleratedHNSWParams parameters) {}
}

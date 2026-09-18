/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CagraIndexParams.HnswHeuristicType;

/**
 * A no-argument {@link Lucene101AcceleratedHNSWCodec} configured through JVM system properties.
 *
 * <p>Maximum-connections property: {@code com.nvidia.cuvs.lucene.hnsw.maxConn}
 *
 * <p>Beam-width property: {@code com.nvidia.cuvs.lucene.hnsw.beamWidth}
 *
 * <p>Set both before construction. Each constructor call snapshots both values, so later property
 * changes affect only later instances. Because JVM system properties are process-global, callers
 * that change them from multiple threads must serialize setting both properties and constructing
 * the codec.
 *
 * <p>This class is intentionally not a Lucene SPI provider. Load it explicitly by class name when
 * a binding can invoke only a public no-argument constructor.
 *
 * @since 26.12
 */
public final class Lucene101ConfiguredHNSWCodec extends Lucene101AcceleratedHNSWCodec {

  /** JVM property containing Lucene's HNSW {@code maxConn} build parameter. */
  public static final String MAX_CONN_PROPERTY = "com.nvidia.cuvs.lucene.hnsw.maxConn";

  /** JVM property containing Lucene's HNSW {@code beamWidth} build parameter. */
  public static final String BEAM_WIDTH_PROPERTY = "com.nvidia.cuvs.lucene.hnsw.beamWidth";

  private final AcceleratedHNSWParams configuredParameters;

  /**
   * Constructs an accelerated HNSW codec from the required JVM system properties.
   *
   * @throws IllegalStateException if either required property is absent
   * @throws IllegalArgumentException if either property is not an integer in its supported range
   * @throws Exception if the delegated codec cannot be constructed
   */
  public Lucene101ConfiguredHNSWCodec() throws Exception {
    this(readConfiguredParameters());
  }

  private Lucene101ConfiguredHNSWCodec(AcceleratedHNSWParams parameters) throws Exception {
    super(parameters);
    configuredParameters = parameters;
  }

  AcceleratedHNSWParams configuredParameters() {
    return configuredParameters;
  }

  private static AcceleratedHNSWParams readConfiguredParameters() {
    int maxConn =
        readRequiredInteger(
            MAX_CONN_PROPERTY,
            AcceleratedHNSWParams.MIN_MAX_CONN,
            AcceleratedHNSWParams.MAX_MAX_CONN);
    int beamWidth =
        readRequiredInteger(
            BEAM_WIDTH_PROPERTY,
            AcceleratedHNSWParams.MIN_BEAM_WIDTH,
            AcceleratedHNSWParams.MAX_BEAM_WIDTH);

    return new AcceleratedHNSWParams.Builder()
        .withStrategy(AcceleratedHNSWParams.Strategy.HEURISTIC)
        .withHnswHeuristicType(HnswHeuristicType.SAME_GRAPH_FOOTPRINT)
        .withMaxConn(maxConn)
        .withBeamWidth(beamWidth)
        .build();
  }

  private static int readRequiredInteger(String propertyName, int minimum, int maximum) {
    String configuredValue = System.getProperty(propertyName);
    if (configuredValue == null) {
      throw new IllegalStateException("Required JVM system property is not set: " + propertyName);
    }

    final int parsedValue;
    try {
      parsedValue = Integer.parseInt(configuredValue);
    } catch (NumberFormatException error) {
      throw new IllegalArgumentException(
          "JVM system property "
              + propertyName
              + " must be an integer, but was: "
              + configuredValue,
          error);
    }

    if (parsedValue < minimum || parsedValue > maximum) {
      throw new IllegalArgumentException(
          "JVM system property "
              + propertyName
              + " must be in range ["
              + minimum
              + ", "
              + maximum
              + "], but was: "
              + configuredValue);
    }
    return parsedValue;
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;

/** Per-build, thread-safe measurements for the bulk CAGRA-to-HNSW pipeline. */
public final class CagraHnswBuildMetrics {

  private final Map<String, StageMeasurement> stages = new ConcurrentHashMap<>();
  private final Map<String, LongAdder> counters = new ConcurrentHashMap<>();
  private final Map<String, Long> gauges = new ConcurrentHashMap<>();

  static long start() {
    return System.nanoTime();
  }

  void stop(String stage, long startedAtNanos) {
    record(stage, System.nanoTime() - startedAtNanos, 0L);
  }

  void stop(String stage, long startedAtNanos, long byteCount) {
    record(stage, System.nanoTime() - startedAtNanos, byteCount);
  }

  void record(String stage, long elapsedNanos, long byteCount) {
    if (elapsedNanos < 0L || byteCount < 0L) {
      throw new IllegalArgumentException("elapsedNanos and byteCount must be non-negative");
    }
    stages.compute(
        stage,
        (ignored, current) ->
            current == null
                ? new StageMeasurement(elapsedNanos, 1L, byteCount)
                : current.plus(elapsedNanos, byteCount));
  }

  void addCounter(String name, long value) {
    counters.computeIfAbsent(name, ignored -> new LongAdder()).add(value);
  }

  /** Records a configuration value that must remain identical across every segment in a build. */
  void setGauge(String name, long value) {
    Objects.requireNonNull(name, "name");
    Long existing = gauges.putIfAbsent(name, value);
    if (existing != null && existing.longValue() != value) {
      throw new IllegalStateException(
          "Gauge \"" + name + "\" changed from " + existing + " to " + value);
    }
  }

  /** Returns a stable machine-readable copy; later measurements do not mutate it. */
  public Map<String, Number> snapshot() {
    Map<String, Number> result = new LinkedHashMap<>();
    stages.keySet().stream()
        .sorted()
        .forEach(
            stage -> {
              StageMeasurement measurement = stages.get(stage);
              String prefix = "stage/" + stage;
              result.put(prefix + "/seconds", measurement.nanos() / 1e9);
              result.put(prefix + "/count", measurement.count());
              if (measurement.bytes() != 0L) {
                result.put(prefix + "/bytes", measurement.bytes());
              }
            });
    counters.keySet().stream()
        .sorted()
        .forEach(name -> result.put("counter/" + name, counters.get(name).sum()));
    gauges.keySet().stream()
        .sorted()
        .forEach(name -> result.put("gauge/" + name, gauges.get(name)));
    return Map.copyOf(result);
  }

  /** Adds this build's snapshot under {@code prefix} to a benchmark result map. */
  public void appendTo(Map<String, Object> target, String prefix) {
    snapshot().forEach((key, value) -> target.put(prefix + "/" + key, value));
  }

  private record StageMeasurement(long nanos, long count, long bytes) {
    StageMeasurement plus(long additionalNanos, long additionalBytes) {
      return new StageMeasurement(nanos + additionalNanos, count + 1L, bytes + additionalBytes);
    }
  }
}

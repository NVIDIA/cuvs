/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.util.Locale;
import org.apache.lucene.util.InfoStream;

/** Reports the graph-processing path selected for one Lucene writer. */
final class GraphProcessingTrace {

  enum Stage {
    MATERIALIZATION,
    SERIALIZATION
  }

  enum Mode {
    SERIAL,
    PARALLEL
  }

  enum Reason {
    ABOVE_THRESHOLD,
    BELOW_THRESHOLD,
    DEVICE_HOST_COPY,
    HOST_SOURCE,
    MEMORY_ADMISSION_DENIED,
    SINGLE_THREAD
  }

  private static final GraphProcessingTrace DISABLED = new GraphProcessingTrace(null, null);

  private final InfoStream infoStream;
  private final String component;

  private GraphProcessingTrace(InfoStream infoStream, String component) {
    this.infoStream = infoStream;
    this.component = component;
  }

  static GraphProcessingTrace disabled() {
    return DISABLED;
  }

  static GraphProcessingTrace toInfoStream(InfoStream infoStream, String component) {
    return new GraphProcessingTrace(infoStream, component);
  }

  void record(Stage stage, Mode mode, Reason reason, int requestedThreads, int nodes) {
    if (infoStream == null || !infoStream.isEnabled(component)) {
      return;
    }
    infoStream.message(component, message(stage, mode, reason, requestedThreads, nodes));
  }

  void recordCopyAdmission(
      Mode mode,
      Reason reason,
      int requestedThreads,
      int nodes,
      long columns,
      long requiredCopyBytes,
      long configuredBudgetBytes) {
    if (infoStream == null || !infoStream.isEnabled(component)) {
      return;
    }
    infoStream.message(
        component,
        message(Stage.MATERIALIZATION, mode, reason, requestedThreads, nodes)
            + " columns="
            + columns
            + " requiredCopyBytes="
            + requiredCopyBytes
            + " configuredBudgetBytes="
            + configuredBudgetBytes);
  }

  private static String message(
      Stage stage, Mode mode, Reason reason, int requestedThreads, int nodes) {
    return "graph-processing stage="
        + lowerCase(stage)
        + " mode="
        + lowerCase(mode)
        + " reason="
        + lowerCase(reason)
        + " requestedThreads="
        + requestedThreads
        + " nodes="
        + nodes;
  }

  private static String lowerCase(Enum<?> value) {
    return value.name().toLowerCase(Locale.ROOT).replace('_', '-');
  }
}

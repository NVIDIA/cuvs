/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import com.nvidia.cuvs.CuVSMatrix;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.HnswGraph.NodesIterator;
import org.junit.Test;

/**
 * Verifies that {@code writerThreads &gt; 1} produces the same result as the serial path for the
 * two parallelizations gated on {@code AcceleratedHNSWUtils}/{@code GPUBuiltHnswGraph}'s {@code
 * PARALLEL_MIN_NODES} threshold: materializing the CAGRA adjacency into {@code NeighborArray}s
 * (the {@code GPUBuiltHnswGraph} constructor), and encoding level 0 to disk ({@code
 * AcceleratedHNSWUtils#writeGraph}).
 *
 * <p>Both tests use a synthetic adjacency ({@link CuVSMatrix#ofArray(int[][])}, the same host-matrix
 * construction the higher-layer subset builder already uses) rather than a real CAGRA build, so
 * that the comparison isolates these two parallelizations from CAGRA's own build-to-build
 * variance -- a real GPU build is not guaranteed to produce the identical graph twice even with the
 * same input and thread count, which would make an end-to-end build comparison unreliable for this
 * purpose. This needs cuVS/GPU only to allocate the host matrix, not to run a build.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestWriterThreadsGraphEquivalence extends LuceneTestCase {

  // Must be >= PARALLEL_MIN_NODES (1 &lt;&lt; 16) in both AcceleratedHNSWUtils and
  // GPUBuiltHnswGraph, or the "parallel" runs below silently fall through to the serial branch and
  // the test would pass without exercising anything.
  private static final int NUM_NODES = (1 << 16) + 1000;
  private static final int DEGREE = 12;
  private static final int NUM_THREADS = 4;

  @Test
  public void fillNeighborArrayParallelMatchesSerial() throws Exception {
    assumeTrue("cuVS not supported", isSupported());
    int[][] adjacency = randomAdjacency(NUM_NODES, DEGREE, new Random(1));

    try (CuVSMatrix matrix = CuVSMatrix.ofArray(adjacency)) {
      GPUBuiltHnswGraph serial = newSingleLayerGraph(matrix, 1);
      GPUBuiltHnswGraph parallel = newSingleLayerGraph(matrix, NUM_THREADS);
      assertGraphsEqual(serial, parallel);
    }
  }

  @Test
  public void writeGraphParallelMatchesSerial() throws Exception {
    assumeTrue("cuVS not supported", isSupported());
    int[][] adjacency = randomAdjacency(NUM_NODES, DEGREE, new Random(2));

    try (CuVSMatrix matrix = CuVSMatrix.ofArray(adjacency);
        Directory dir = new ByteBuffersDirectory()) {
      // Materialize once, serially, so any difference found below is attributable only to
      // writeGraph's own parallelization, not to fillNeighborArray's.
      GPUBuiltHnswGraph graph = newSingleLayerGraph(matrix, 1);

      int[][] serialOffsets;
      try (IndexOutput out = dir.createOutput("serial", IOContext.DEFAULT)) {
        serialOffsets = AcceleratedHNSWUtils.writeGraph(graph, out, 1);
      }
      int[][] parallelOffsets;
      try (IndexOutput out = dir.createOutput("parallel", IOContext.DEFAULT)) {
        parallelOffsets = AcceleratedHNSWUtils.writeGraph(graph, out, NUM_THREADS);
      }

      assertEquals(serialOffsets.length, parallelOffsets.length);
      for (int level = 0; level < serialOffsets.length; level++) {
        assertArrayEquals(
            "per-node byte-length offsets differ for level " + level,
            serialOffsets[level],
            parallelOffsets[level]);
      }

      assertArrayEquals(
          "writeGraph's parallel level-0 encoding produced different bytes than the serial path",
          readAllBytes(dir, "serial"),
          readAllBytes(dir, "parallel"));
    }
  }

  private static GPUBuiltHnswGraph newSingleLayerGraph(CuVSMatrix layer0Adjacency, int numThreads)
      throws IOException {
    // A single layer (layer 0 only): the constructor never consults layerNodes in that case, so
    // the placeholder null entry mirrors the convention used elsewhere for "layer 0 needs no node
    // list" without actually being read.
    return new GPUBuiltHnswGraph(
        NUM_NODES,
        /* dimensions= */ 4,
        Arrays.asList((int[]) null),
        List.of(layer0Adjacency),
        numThreads);
  }

  /** Every node/level's in-order arc list must match exactly between the two graphs. */
  private static void assertGraphsEqual(HnswGraph a, HnswGraph b) throws Exception {
    assertEquals(a.numLevels(), b.numLevels());
    for (int level = 0; level < a.numLevels(); level++) {
      int[] nodes = NodesIterator.getSortedNodes(a.getNodesOnLevel(level));
      for (int node : nodes) {
        assertArrayEquals(
            "node " + node + " at level " + level + " has different neighbors",
            arcsOf(a, level, node),
            arcsOf(b, level, node));
      }
    }
  }

  private static int[] arcsOf(HnswGraph graph, int level, int node) throws Exception {
    graph.seek(level, node);
    List<Integer> arcs = new ArrayList<>();
    for (int n = graph.nextNeighbor(); n != NO_MORE_DOCS; n = graph.nextNeighbor()) {
      arcs.add(n);
    }
    return arcs.stream().mapToInt(Integer::intValue).toArray();
  }

  private static byte[] readAllBytes(Directory dir, String name) throws Exception {
    try (IndexInput in = dir.openInput(name, IOContext.DEFAULT)) {
      byte[] bytes = new byte[(int) in.length()];
      in.readBytes(bytes, 0, bytes.length);
      return bytes;
    }
  }

  /**
   * A deterministic, seeded pseudo-adjacency. It doesn't need to be a real CAGRA graph -- only a
   * realistic shape (fixed degree, valid node ids) -- since {@code GPUBuiltHnswGraph} and {@code
   * AcceleratedHNSWUtils#writeGraph} don't interpret the neighbor ids semantically.
   */
  private static int[][] randomAdjacency(int numNodes, int degree, Random random) {
    int[][] adjacency = new int[numNodes][degree];
    for (int[] row : adjacency) {
      for (int j = 0; j < degree; j++) {
        row[j] = random.nextInt(numNodes);
      }
    }
    return adjacency;
  }
}

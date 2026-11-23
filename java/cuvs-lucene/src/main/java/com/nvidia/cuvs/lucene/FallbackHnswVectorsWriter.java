/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.Closeable;
import java.io.IOException;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter.MergedVectorValues;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.util.hnsw.HnswGraph;

/**
 * Lucene's HNSW writer, which the accelerated-HNSW formats fall back to without a GPU, made able to
 * merge segments whose graph records an M other than its own.
 *
 * <p>Lucene's writer starts the merged graph from the graphs of the source segments, and assumes
 * that they record its own M. A graph that records another M breaks the merge:
 *
 * <ul>
 *   <li>a smaller M fails it with "No growth is allowed" since Lucene 10.4, which sizes the merged
 *       graph for the M its largest source graph recorded, once a node gets more neighbors than
 *       that M allows; up to Lucene 10.3, the nodes of the source graph keep their fewer neighbors
 *       in the merged graph;
 *   <li>a larger M can fail it the same way up to Lucene 10.3, which copies the neighbors of the
 *       source graph into a graph sized for the writer's M, and since Lucene 10.4 can give a
 *       merged segment with more neighbors per node than the M it records lets a reader read.
 * </ul>
 *
 * <p>cuvs-lucene 26.10 and earlier recorded half the degree of the graph cuVS built: less than maxConn
 * in a segment with fewer vectors than the graph degree, where cuVS lowers the degree, and in every
 * segment built with the {@code SIMILAR_SEARCH_PERFORMANCE} heuristic or a {@code CUSTOM} graph
 * degree below {@code 2 * maxConn}, and more than maxConn with a {@code CUSTOM} graph degree above
 * {@code 2 * maxConn}, which later releases reject. Later releases record maxConn, but segments
 * written earlier keep their M, as do segments written with another maxConn, or by Lucene's own
 * format with another M.
 *
 * <p>So while Lucene's writer builds the merged graph of a field, every source segment whose graph
 * records another M is replaced in the {@link MergeState} by a view that is not an {@link
 * HnswGraphProvider}, and Lucene adds the vectors of that segment to the merged graph one by one, as
 * it does for a segment without a graph. Lucene merges the fields one at a time, so the change to
 * the {@code MergeState} is undone before the next field uses it.
 *
 * <p>Lucene 11 will not need this: on Lucene's {@code main} branch, since apache/lucene#14121, a
 * merge builds and records the merged graph with the M of the graph it starts from, so another M
 * no longer breaks it, although the merged segment then keeps that M rather than the writer's. The
 * 10.x branches never got that change. Once cuvs-lucene supports a Lucene release that has it,
 * remove this class from the modules of those releases (see {@code MAINTAINING.md}).
 */
final class FallbackHnswVectorsWriter extends CompatFilterKnnVectorsWriter {

  private final int maxConn;

  /**
   * Returns Lucene's HNSW writer, made able to merge segments whose graph records another M, for
   * the CPU fallback of an accelerated-HNSW format: it builds graphs with the HNSW parameters of
   * {@code params}, over {@code flatWriter}, which {@code flatFormat} created.
   */
  static KnnVectorsWriter create(
      SegmentWriteState state,
      AcceleratedHNSWParams params,
      FlatVectorsFormat flatFormat,
      FlatVectorsWriter flatWriter)
      throws IOException {
    return new FallbackHnswVectorsWriter(
        LuceneCompat.hnswVectorsWriter(
            state,
            params.getMaxConn(),
            params.getBeamWidth(),
            flatFormat,
            flatWriter,
            params.getNumMergeWorkers(),
            new TaskExecutor(params.getMergeExec())),
        params.getMaxConn());
  }

  private FallbackHnswVectorsWriter(KnnVectorsWriter in, int maxConn) {
    super(in);
    this.maxConn = maxConn;
  }

  @Override
  Closeable prepareGraphMerge(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    KnnVectorsReader[] readers = mergeState.knnVectorsReaders;
    KnnVectorsReader[] originals = readers.clone();
    Closeable undo = () -> System.arraycopy(originals, 0, readers, 0, readers.length);
    try {
      for (int i = 0; i < readers.length; i++) {
        // The segments that Lucene's writer passes to its graph merger.
        if (MergedVectorValues.hasVectorValues(mergeState.fieldInfos[i], fieldInfo.name) == false) {
          continue;
        }
        HnswGraph graph = graphOf(readers[i], fieldInfo.name);
        if (graph != null && graph.size() > 0 && graph.maxConn() != maxConn) {
          readers[i] = new WithoutGraphs(readers[i]);
        }
      }
    } catch (Throwable t) {
      undo.close();
      throw t;
    }
    return undo;
  }

  /** Returns the graph {@code reader} holds for {@code field}, or null if it holds none. */
  private static HnswGraph graphOf(KnnVectorsReader reader, String field) throws IOException {
    if (reader instanceof PerFieldKnnVectorsFormat.FieldsReader perField) {
      reader = perField.getFieldReader(field);
    }
    return reader instanceof HnswGraphProvider provider ? provider.getGraph(field) : null;
  }

  /** A source segment of a merge, without the graphs that Lucene would merge from. */
  private static final class WithoutGraphs extends CompatFilterKnnVectorsReader {

    WithoutGraphs(KnnVectorsReader in) {
      super(in);
    }
  }
}

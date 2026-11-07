/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.util.Map;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.util.Bits;

/**
 * Base class of the GPU search reader, {@code CuVS2510GPUVectorsReader}, that hides a signature
 * change in {@link KnnVectorsReader}. Since Lucene 10.3, the documents a search may return come as
 * {@link AcceptDocs}, where Lucene 10.2 passed {@code Bits}. Each variant of this class overrides
 * the {@code search} methods of its Lucene release and forwards them to {@code doSearch}, whose
 * signature is the same on every release, so the shared reader implements {@code doSearch} once.
 *
 * <p>This variant converts with {@code AcceptDocs.bits()} and reports the reader's off-heap size,
 * as the Lucene 10.3 variant does. Since Lucene 10.6, the merge instance of the raw vectors reader
 * reads the file through an input of its own, unless the one searches use suits a merge, instead of
 * changing how searches read it, so this variant has merges read through a reader created around
 * it, as Lucene's HNSW reader does. It also passes the vector counts and sequential reads Lucene
 * 10.6 added on to the raw vectors reader.
 */
abstract class CompatKnnVectorsReader extends KnnVectorsReader {

  private final FlatVectorsReader rawVectorsReader;

  /**
   * @param rawVectorsReader reads the raw vectors of the fields
   */
  CompatKnnVectorsReader(FlatVectorsReader rawVectorsReader) {
    this.rawVectorsReader = rawVectorsReader;
  }

  /**
   * Reports only the raw vectors. Lucene asks for the memory that searches need to avoid page
   * faults; searches read the raw vectors from their file, but search the cuVS indexes on the GPU.
   * The file holding the cuVS indexes is only read to load them onto the GPU, so it is left out,
   * even though it stays open.
   */
  @Override
  public Map<String, Long> getOffHeapByteSize(FieldInfo fieldInfo) {
    return rawVectorsReader.getOffHeapByteSize(fieldInfo);
  }

  /** Returns a reader that reads the raw vectors through their reader's merge instance. */
  @Override
  public KnnVectorsReader getMergeInstance() throws IOException {
    FlatVectorsReader mergeInstance = rawVectorsReader.getMergeInstance();
    return mergeInstance == rawVectorsReader ? this : withRawVectorsReader(mergeInstance);
  }

  /** Lets the raw vectors reader release what it held for the merge. */
  @Override
  public void finishMerge() throws IOException {
    rawVectorsReader.finishMerge();
  }

  /** Reads the raw vectors, which are the float vectors of the fields, as {@code access} says. */
  @Override
  public FloatVectorValues getFloatVectorValues(String field, DataAccessHint access)
      throws IOException {
    return rawVectorsReader.getFloatVectorValues(field, access);
  }

  /** Reads the number of vectors from the metadata of the raw vectors. */
  @Override
  public int getVectorCount(FieldInfo fieldInfo) throws IOException {
    return rawVectorsReader.getVectorCount(fieldInfo);
  }

  @Override
  public void search(String field, float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs)
      throws IOException {
    doSearch(field, target, knnCollector, acceptDocs == null ? null : acceptDocs.bits());
  }

  @Override
  public void search(String field, byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs)
      throws IOException {
    doSearch(field, target, knnCollector, acceptDocs == null ? null : acceptDocs.bits());
  }

  /**
   * Implements {@link #search(String, float[], KnnCollector, AcceptDocs)}.
   *
   * @param acceptDocs the documents that may be returned, or {@code null} if all may be
   */
  protected abstract void doSearch(
      String field, float[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException;

  /**
   * Implements {@link #search(String, byte[], KnnCollector, AcceptDocs)}.
   *
   * @param acceptDocs the documents that may be returned, or {@code null} if all may be
   */
  protected abstract void doSearch(
      String field, byte[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException;

  /**
   * Returns a reader for a merge that reads the raw vectors through {@code rawVectorsReader}, the
   * merge instance of this reader's, and shares everything else with this reader. Like Lucene's
   * merge instances, it is used only by the merge thread, and Lucene never closes it; closing it
   * does nothing, since this reader owns what they share.
   */
  abstract KnnVectorsReader withRawVectorsReader(FlatVectorsReader rawVectorsReader);
}

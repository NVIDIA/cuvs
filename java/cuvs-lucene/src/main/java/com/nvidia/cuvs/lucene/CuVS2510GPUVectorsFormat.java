/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.assertIsSupported;

import com.nvidia.cuvs.LibraryException;
import java.io.IOException;
import java.util.logging.Logger;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

/**
 * Extends upon the KnnVectorsFormat - Encodes/decodes per-document vector and any associated indexing structures required to support
 * GPU-based accelerated nearest-neighbor search.
 *
 * @since 25.10
 */
public class CuVS2510GPUVectorsFormat extends KnnVectorsFormat {

  @SuppressWarnings("unused")
  private static final Logger log = Logger.getLogger(CuVS2510GPUVectorsFormat.class.getName());

  private static final int maxDimensions = 4096;
  private static final LuceneProvider LUCENE_PROVIDER;
  private static final FlatVectorsFormat FLAT_VECTORS_FORMAT;
  private GPUSearchParams gpuSearchParams;

  static final String CUVS_META_CODEC_NAME = "Lucene102CuVSVectorsFormatMeta";
  static final String CUVS_META_CODEC_EXT = "vemc";
  static final String CUVS_INDEX_CODEC_NAME = "Lucene102CuVSVectorsFormatIndex";
  static final String CUVS_INDEX_EXT = "vcag";
  static final int VERSION_START = 0;
  static final int VERSION_CURRENT = VERSION_START;

  static {
    try {
      LUCENE_PROVIDER = LuceneProvider.getInstance("99");
      FLAT_VECTORS_FORMAT =
          LUCENE_PROVIDER.getLuceneFlatVectorsFormatInstance(DefaultFlatVectorScorer.INSTANCE);
    } catch (Exception e) {
      throw new ExceptionInInitializerError(e.getMessage());
    }
  }

  /**
   * Initializes the {@link CuVS2510GPUVectorsFormat} with default parameter values.
   *
   * @throws LibraryException if the native library fails to load
   */
  public CuVS2510GPUVectorsFormat() {
    this(new GPUSearchParams.Builder().build());
  }

  /**
   * Initializes the {@link CuVS2510GPUVectorsFormat} with the given threads, graph degree, etc.
   *
   * @param gpuSearchParams An instance of {@link GPUSearchParams}
   * @throws LibraryException if the native library fails to load
   */
  public CuVS2510GPUVectorsFormat(GPUSearchParams gpuSearchParams) {
    super("CuVS2510GPUVectorsFormat");
    this.gpuSearchParams = gpuSearchParams;
  }

  /**
   * Returns a {@link CuVS2510GPUVectorsWriter} to write the vectors to the index.
   */
  @Override
  public CuVS2510GPUVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    assertIsSupported();
    var flatWriter = FLAT_VECTORS_FORMAT.fieldsWriter(state);
    return new CuVS2510GPUVectorsWriter(state, gpuSearchParams, flatWriter);
  }

  /**
   * Returns a KnnVectorsReader to read the vectors from the index.
   */
  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    assertIsSupported();
    return new CuVS2510GPUVectorsReader(state, FLAT_VECTORS_FORMAT.fieldsReader(state));
  }

  /**
   * Returns the maximum number of vector dimensions supported by this codec for the given field name.
   */
  @Override
  public int getMaxDimensions(String fieldName) {
    return maxDimensions;
  }
}

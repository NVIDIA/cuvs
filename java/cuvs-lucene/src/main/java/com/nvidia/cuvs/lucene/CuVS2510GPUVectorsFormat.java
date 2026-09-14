/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.assertIsSupported;

import java.io.IOException;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
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

  private static final int MAX_DIMENSIONS = 4096;
  private static volatile FlatVectorsFormat cachedFlatVectorsFormat;

  public static final String CUVS_META_CODEC_NAME = "Lucene102CuVSVectorsFormatMeta";
  public static final String CUVS_META_CODEC_EXT = "vemc";
  public static final String CUVS_INDEX_CODEC_NAME = "Lucene102CuVSVectorsFormatIndex";
  public static final String CUVS_INDEX_EXT = "vcag";
  public static final int VERSION_START = 0;
  public static final int VERSION_CURRENT = VERSION_START;

  private final GPUSearchParams gpuSearchParams;
  private final FilterBitsetCache filterBitsetCache;

  private static LuceneProvider getLucene99Provider() throws IOException {
    try {
      return LuceneProvider.getInstance(LuceneProvider.LUCENE_99_FORMAT_VERSION);
    } catch (ClassNotFoundException e) {
      throw new IOException("Lucene99 vector formats are not available in this runtime", e);
    }
  }

  private static FlatVectorsFormat getOrCreateFlatVectorsFormat() throws IOException {
    FlatVectorsFormat format = cachedFlatVectorsFormat;
    if (format == null) {
      synchronized (CuVS2510GPUVectorsFormat.class) {
        format = cachedFlatVectorsFormat;
        if (format == null) {
          try {
            format =
                getLucene99Provider()
                    .getLuceneFlatVectorsFormatInstance(DefaultFlatVectorScorer.INSTANCE);
            cachedFlatVectorsFormat = format;
          } catch (Exception e) {
            throw Utils.handleThrowable(e);
          }
        }
      }
    }
    return format;
  }

  /**
   * Initializes the {@link CuVS2510GPUVectorsFormat} with default parameter values.
   */
  public CuVS2510GPUVectorsFormat() {
    this(new GPUSearchParams.Builder().build(), FilterBitsetCacheConfig.DEFAULT);
  }

  /**
   * Initializes the {@link CuVS2510GPUVectorsFormat} with an instance of {@link GPUSearchParams}.
   *
   * @param gpuSearchParams An instance of {@link GPUSearchParams}
   */
  public CuVS2510GPUVectorsFormat(GPUSearchParams gpuSearchParams) {
    this(gpuSearchParams, FilterBitsetCacheConfig.DEFAULT);
  }

  /**
   * Initializes the format with GPU search and filter-bitset-cache parameters.
   *
   * @param gpuSearchParams GPU index and search parameters
   * @param filterCacheConfig filter-bitset-cache configuration
   */
  public CuVS2510GPUVectorsFormat(
      GPUSearchParams gpuSearchParams, FilterBitsetCacheConfig filterCacheConfig) {
    super("CuVS2510GPUVectorsFormat");
    this.gpuSearchParams = gpuSearchParams;
    this.filterBitsetCache = new FilterBitsetCache(filterCacheConfig);
  }

  /**
   * Returns a KnnVectorsWriter instance to write the vectors to the index.
   */
  @Override
  public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    assertIsSupported();
    var flatWriter = getOrCreateFlatVectorsFormat().fieldsWriter(state);
    return new CuVS2510GPUVectorsWriter(state, gpuSearchParams, flatWriter);
  }

  /**
   * Returns a KnnVectorsReader instance to read the vectors from the index.
   */
  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    assertIsSupported();
    return new CuVS2510GPUVectorsReader(
        state, getOrCreateFlatVectorsFormat().fieldsReader(state), filterBitsetCache);
  }

  /**
   * Returns the maximum number of vector dimensions supported by this codec for the given field name.
   */
  @Override
  public int getMaxDimensions(String fieldName) {
    return MAX_DIMENSIONS;
  }
}

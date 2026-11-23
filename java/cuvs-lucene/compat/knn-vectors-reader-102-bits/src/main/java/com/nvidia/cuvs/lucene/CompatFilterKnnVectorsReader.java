/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.util.Bits;

/**
 * The part of {@link FilterKnnVectorsReader} whose signatures differ between Lucene releases:
 * Lucene 10.2 passes the documents a search may return as {@link Bits}; Lucene 10.3 changed that
 * parameter to {@code AcceptDocs}.
 */
abstract class CompatFilterKnnVectorsReader extends FilterKnnVectorsReader {

  /**
   * @param in the reader to forward to
   */
  CompatFilterKnnVectorsReader(KnnVectorsReader in) {
    super(in);
  }

  @Override
  public void search(String field, float[] target, KnnCollector knnCollector, Bits acceptDocs)
      throws IOException {
    in.search(field, target, knnCollector, acceptDocs);
  }

  @Override
  public void search(String field, byte[] target, KnnCollector knnCollector, Bits acceptDocs)
      throws IOException {
    in.search(field, target, knnCollector, acceptDocs);
  }
}

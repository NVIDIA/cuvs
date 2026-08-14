/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.bench;

import com.nvidia.cuvs.lucene.Lucene101AcceleratedHNSWCodec;
import java.io.IOException;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

/** Test-only codec that reports the writer selected by the production HNSW codec. */
public final class PyLuceneWriterSelectionCodec extends Lucene101AcceleratedHNSWCodec {

  public PyLuceneWriterSelectionCodec() throws Exception {
    super();
    setKnnFormat(new WriterSelectionFormat(super.knnVectorsFormat()));
  }

  private static final class WriterSelectionFormat extends KnnVectorsFormat {

    private static final String NOT_SELECTED = "not-selected";

    private final KnnVectorsFormat delegate;
    private volatile String writerClass = NOT_SELECTED;

    private WriterSelectionFormat(KnnVectorsFormat delegate) {
      super(delegate.getName());
      this.delegate = delegate;
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
      KnnVectorsWriter writer = delegate.fieldsWriter(state);
      String selectedClass = writer.getClass().getName();
      String previousClass = writerClass;
      if (!NOT_SELECTED.equals(previousClass) && !previousClass.equals(selectedClass)) {
        throw new AssertionError(
            "Vector writer selection changed from " + previousClass + " to " + selectedClass);
      }
      writerClass = selectedClass;
      return writer;
    }

    @Override
    public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
      return delegate.fieldsReader(state);
    }

    @Override
    public int getMaxDimensions(String fieldName) {
      return delegate.getMaxDimensions(fieldName);
    }

    @Override
    public String toString() {
      return getName() + "(writerClass=" + writerClass + ")";
    }
  }
}

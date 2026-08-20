/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.bench;

import java.io.IOException;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

/** Test-only codec that reports the writer selected by the configured production HNSW codec. */
public final class PyLuceneWriterSelectionCodec extends FilterCodec {

  private static final String CONFIGURED_CODEC_CLASS =
      "com.nvidia.cuvs.bench.PyLuceneConfiguredHnswCodec";

  private final KnnVectorsFormat knnVectorsFormat;
  private final String configuredCodecDiagnostics;

  public PyLuceneWriterSelectionCodec() throws Exception {
    this(configuredCodec());
  }

  private PyLuceneWriterSelectionCodec(Codec delegate) {
    super(delegate.getName(), delegate);
    knnVectorsFormat = new WriterSelectionFormat(delegate.knnVectorsFormat());
    configuredCodecDiagnostics = delegate.toString();
  }

  private static Codec configuredCodec() throws Exception {
    return (Codec)
        Class.forName(CONFIGURED_CODEC_CLASS).getConstructor().newInstance();
  }

  @Override
  public KnnVectorsFormat knnVectorsFormat() {
    return knnVectorsFormat;
  }

  @Override
  public String toString() {
    return getClass().getSimpleName() + "(" + configuredCodecDiagnostics + ")";
  }

  private static final class WriterSelectionFormat extends KnnVectorsFormat {

    private static final String NOT_SELECTED = "not-selected";

    private final KnnVectorsFormat delegate;
    private String writerClass = NOT_SELECTED;
    private int fieldsWriterCalls;

    private WriterSelectionFormat(KnnVectorsFormat delegate) {
      super(delegate.getName());
      this.delegate = delegate;
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
      KnnVectorsWriter writer = delegate.fieldsWriter(state);
      recordWriter(writer.getClass().getName());
      return writer;
    }

    private synchronized void recordWriter(String selectedClass) {
      if (!NOT_SELECTED.equals(writerClass) && !writerClass.equals(selectedClass)) {
        throw new AssertionError(
            "Vector writer selection changed from " + writerClass + " to " + selectedClass);
      }
      writerClass = selectedClass;
      fieldsWriterCalls++;
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
    public synchronized String toString() {
      return getName()
          + "(writerClass="
          + writerClass
          + ", fieldsWriterCalls="
          + fieldsWriterCalls
          + ")";
    }
  }
}

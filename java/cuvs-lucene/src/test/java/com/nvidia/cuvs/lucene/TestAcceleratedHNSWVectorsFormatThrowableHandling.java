/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.util.Map;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.junit.Test;

public class TestAcceleratedHNSWVectorsFormatThrowableHandling extends LuceneTestCase {

  @Test
  public void testReadersRethrowIOExceptionUnchanged() throws Exception {
    assertReaderFormatsRethrowUnchanged(new IOException("reader I/O failure"));
  }

  @Test
  public void testReadersRethrowRuntimeExceptionUnchanged() throws Exception {
    assertReaderFormatsRethrowUnchanged(new IllegalStateException("reader runtime failure"));
  }

  @Test
  public void testReadersRethrowErrorUnchanged() throws Exception {
    assertReaderFormatsRethrowUnchanged(new AssertionError("reader error"));
  }

  private void assertReaderFormatsRethrowUnchanged(Throwable failure) throws Exception {
    for (KnnVectorsFormat format : readerFormats()) {
      try (Directory directory = new ThrowingDirectory(failure)) {
        SegmentReadState state = newSegmentReadState(directory);
        Throwable thrown = assertThrows(failure.getClass(), () -> format.fieldsReader(state));
        assertSame(format.getName(), failure, thrown);
      }
    }
  }

  private static KnnVectorsFormat[] readerFormats() {
    return new KnnVectorsFormat[] {
      new Lucene99AcceleratedHNSWVectorsFormat(),
      new LuceneAcceleratedHNSWScalarQuantizedVectorsFormat(),
      new LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat()
    };
  }

  private static SegmentReadState newSegmentReadState(Directory directory) {
    SegmentInfo segmentInfo =
        new SegmentInfo(
            directory,
            Version.LATEST,
            Version.LATEST,
            "_0",
            0,
            false,
            false,
            Codec.getDefault(),
            Map.of(),
            StringHelper.randomId(),
            Map.of(),
            null);
    return new SegmentReadState(directory, segmentInfo, FieldInfos.EMPTY, IOContext.DEFAULT);
  }

  private static final class ThrowingDirectory extends FilterDirectory {
    private final Throwable failure;

    private ThrowingDirectory(Throwable failure) {
      super(new ByteBuffersDirectory());
      this.failure = failure;
    }

    @Override
    public IndexInput openInput(String name, IOContext context) throws IOException {
      if (failure instanceof IOException ioe) {
        throw ioe;
      }
      if (failure instanceof RuntimeException runtimeException) {
        throw runtimeException;
      }
      if (failure instanceof Error error) {
        throw error;
      }
      throw new AssertionError("unexpected test throwable", failure);
    }
  }
}

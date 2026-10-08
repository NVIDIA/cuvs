/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.carrotsearch.randomizedtesting.annotations.Name;
import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.IOConsumer;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.InfoStream;

/** Observes the real codec writer, not IndexWriter's cached buffering counters. Requires cuVS. */
public class TestAcceleratedHNSWHostInputMemory extends LuceneTestCase {
  private static final int VECTORS_PER_SEGMENT = 256;
  private static final String[] FIELDS = {"embedding", "secondary_embedding"};
  private final KnnVectorsFormat format;
  private final int dimensions;
  private final int payloadBytesPerVector;

  public TestAcceleratedHNSWHostInputMemory(
      @Name("format") KnnVectorsFormat format,
      @Name("dimensions") int dimensions,
      @Name("payloadBytesPerVector") int payloadBytesPerVector) {
    this.format = format;
    this.dimensions = dimensions;
    this.payloadBytesPerVector = payloadBytesPerVector;
  }

  @ParametersFactory
  public static List<Object[]> parameters() {
    AcceleratedHNSWParams params =
        new AcceleratedHNSWParams.Builder()
            .withStrategy(AcceleratedHNSWParams.Strategy.CUSTOM)
            .withGraphDegree(32)
            .withIntermediateGraphDegree(64)
            .withHNSWLayer(1)
            .build();
    return Arrays.asList(
        new Object[][] {
          {new Lucene99AcceleratedHNSWVectorsFormat(params), 128, 512},
          {new LuceneAcceleratedHNSWScalarQuantizedVectorsFormat(params), 128, 128},
          // 129 binary dimensions deliberately exercise the partial final byte.
          {new LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat(params), 129, 17}
        });
  }

  public void testFlushAndMergeCountEachFieldsNativeInput() throws Exception {
    AccountingObserver observer = new AccountingObserver();
    try (Directory directory = newDirectory();
        IndexWriter writer = new IndexWriter(directory, config(observer))) {
      addDocuments(writer, 0, VECTORS_PER_SEGMENT);
      writer.commit();
      addDocuments(writer, VECTORS_PER_SEGMENT, VECTORS_PER_SEGMENT);
      writer.commit();
      try (DirectoryReader reader = DirectoryReader.open(writer)) {
        assertEquals(2, reader.leaves().size());
      }
      writer.getConfig().setMergePolicy(new TieredMergePolicy());
      writer.forceMerge(1);
      writer.commit();

      long segmentBytes = (long) VECTORS_PER_SEGMENT * payloadBytesPerVector;
      List<Long> bytesPerSegment = List.of(segmentBytes, segmentBytes, 2 * segmentBytes);
      assertEquals(bytesPerSegment.size(), observer.segments.size());
      List<InputAllocation> expected = new ArrayList<>();
      for (int segment = 0; segment < bytesPerSegment.size(); segment++) {
        for (String field : FIELDS) {
          expected.add(
              new InputAllocation(
                  observer.segments.get(segment), field, bytesPerSegment.get(segment)));
        }
      }
      // Field traversal order is not a contract; identity, size, and exactly-once reporting are.
      assertEquals(expected.size(), observer.allocations.size());
      assertEquals(Set.copyOf(expected), Set.copyOf(observer.allocations));
      try (DirectoryReader reader = DirectoryReader.open(writer)) {
        assertEquals(1, reader.leaves().size());
        assertEquals(2 * VECTORS_PER_SEGMENT, reader.numDocs());
        for (String field : FIELDS) {
          assertEquals(
              reader.numDocs(), getOnlyLeafReader(reader).getFloatVectorValues(field).size());
        }
      }
      TestUtil.checkIndex(directory);
    }
  }

  public void testSingletonFieldsDoNotAllocatePrimaryHostInput() throws Exception {
    AccountingObserver observer = new AccountingObserver();
    try (Directory directory = newDirectory();
        IndexWriter writer = new IndexWriter(directory, config(observer))) {
      addDocuments(writer, 0, 1);
      writer.commit();
      assertTrue(observer.allocations.isEmpty());
      assertTrue("The accelerated writer must actually flush", observer.completedWrites > 0);
    }
  }

  private IndexWriterConfig config(AccountingObserver observer) {
    assumeTrue("cuVS not supported", ThreadLocalCuVSResourcesProvider.isSupported());
    return newIndexWriterConfig()
        .setCodec(TestUtil.alwaysKnnVectorsFormat(observingFormat(observer)))
        .setInfoStream(observer)
        .setMergePolicy(NoMergePolicy.INSTANCE)
        .setMergeScheduler(new SerialMergeScheduler())
        .setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH)
        .setRAMBufferSizeMB(256);
  }

  private void addDocuments(IndexWriter writer, int firstId, int count) throws IOException {
    for (int id = firstId; id < firstId + count; id++) {
      Document document = new Document();
      for (String field : FIELDS) {
        float[] vector = new float[dimensions];
        for (int dimension = 0; dimension < dimensions; dimension++) {
          vector[dimension] = (float) Math.sin((id + 1.0) * (dimension + 1.0));
        }
        document.add(new KnnFloatVectorField(field, vector));
      }
      writer.addDocument(document);
    }
  }

  private KnnVectorsFormat observingFormat(AccountingObserver observer) {
    return new KnnVectorsFormat(format.getName()) {
      @Override
      public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        KnnVectorsWriter delegate = format.fieldsWriter(state);
        if (!(delegate instanceof Lucene99AcceleratedHNSWVectorsWriter)
            && !(delegate instanceof LuceneAcceleratedHNSWScalarQuantizedVectorsWriter)
            && !(delegate instanceof LuceneAcceleratedHNSWBinaryQuantizedVectorsWriter)) {
          IOUtils.closeWhileHandlingException(delegate);
          throw new AssertionError("CPU fallback must not satisfy accounting tests");
        }
        // Capture identities independently of the diagnostic text being checked.
        observer.segments.add(state.segmentInfo.name);
        return new ObservedWriter(delegate, observer, state.segmentInfo.name);
      }

      @Override
      public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return format.fieldsReader(state);
      }

      @Override
      public int getMaxDimensions(String fieldName) {
        return format.getMaxDimensions(fieldName);
      }
    };
  }

  private record InputAllocation(String segment, String field, long bytes) {}

  private static final class AccountingObserver extends InfoStream {
    private final List<InputAllocation> allocations = new ArrayList<>();
    private final List<String> segments = new ArrayList<>();
    private ObservedWriter active;
    private long baseline;
    private int completedWrites;

    @Override
    public void message(String component, String message) {
      if (!message.startsWith("primary_host_input_bytes=")) {
        return;
      }
      assertNotNull("Allocation must occur inside flush or merge", active);
      long bytes =
          Long.parseLong(message.substring(message.indexOf('=') + 1, message.indexOf(' ')));
      assertTrue(message.contains(" segment=" + active.segment + " field="));
      String field = message.substring(message.indexOf(" field=") + " field=".length());
      assertEquals(
          "Codec accounting must include this allocation exactly once",
          baseline + bytes,
          active.delegate.ramBytesUsed());
      allocations.add(new InputAllocation(active.segment, field, bytes));
    }

    @Override
    public boolean isEnabled(String component) {
      return true;
    }

    @Override
    public void close() {}
  }

  /** Captures a baseline immediately before each synchronous codec operation. */
  private static final class ObservedWriter extends KnnVectorsWriter {
    private final KnnVectorsWriter delegate;
    private final AccountingObserver observer;
    private final String segment;

    ObservedWriter(KnnVectorsWriter delegate, AccountingObserver observer, String segment) {
      this.delegate = delegate;
      this.observer = observer;
      this.segment = segment;
    }

    private void observe(IOConsumer<KnnVectorsWriter> operation) throws IOException {
      observer.active = this;
      observer.baseline = delegate.ramBytesUsed();
      operation.accept(delegate);
      assertEquals(
          "Completed scope must not retain native input bytes",
          observer.baseline,
          delegate.ramBytesUsed());
      observer.completedWrites++;
      observer.active = null;
    }

    @Override
    public KnnFieldVectorsWriter<?> addField(FieldInfo info) throws IOException {
      return delegate.addField(info);
    }

    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
      observe(writer -> writer.flush(maxDoc, sortMap));
    }

    @Override
    public void mergeOneField(FieldInfo info, MergeState state) throws IOException {
      observe(writer -> writer.mergeOneField(info, state));
    }

    @Override
    public void finish() throws IOException {
      delegate.finish();
    }

    @Override
    public void close() throws IOException {
      delegate.close();
    }

    @Override
    public long ramBytesUsed() {
      return delegate.ramBytesUsed();
    }
  }
}

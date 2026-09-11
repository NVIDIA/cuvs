/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.Map;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.junit.After;

public class TestExternalFbinFlatVectorsReader extends LuceneTestCase {

  private static final int[] GENERALIZATION_DIMENSIONS = {64, 65, 128, 129, 192, 193, 1536, 2048};
  private static final long FORCED_MMAP_CHUNK_SIZE = 32L;

  @After
  public void clearRegistry() {
    ExternalFbinFileRegistry.clearForTests();
  }

  public void testDescriptorVectorValuesScoringAndIntegrity() throws Exception {
    float[][] vectors = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}};
    Path fbin = writeFbin(createTempDir().resolve("vectors.fbin"), vectors);
    String digest = sha256(fbin);

    try (Directory indexDirectory = newDirectory();
        ExternalFbinFileRegistry.Registration registration =
            ExternalFbinFileRegistry.register(fbin, digest)) {
      ExternalFbinReference reference = registration.reference(0, vectors.length);
      SegmentFixture fixture = writeDescriptor(indexDirectory, reference);

      try (ExternalFbinFlatVectorsReader reader =
          new ExternalFbinFlatVectorsReader(fixture.readState(), 16L)) {
        FloatVectorValues values = reader.getFloatVectorValues("vector");
        assertEquals(vectors.length, values.size());
        assertEquals(vectors[0].length, values.dimension());
        assertArrayEquals(vectors[0], values.vectorValue(0), 0.0f);
        assertArrayEquals(vectors[2], values.copy().vectorValue(2), 0.0f);

        RandomVectorScorer scorer = reader.getRandomVectorScorer("vector", vectors[1]);
        assertEquals(1.0f, scorer.score(1), 0.0f);
        assertTrue(scorer.score(0) < scorer.score(1));
        assertEquals(1, scorer.ordToDoc(1));
        reader.checkIntegrity();
        reader.checkIntegrity();
        assertArrayEquals(vectors[2], reader.getFloatVectorValues("vector").vectorValue(2), 0.0f);

        try (FileChannel channel = FileChannel.open(fbin, StandardOpenOption.WRITE)) {
          channel.write(ByteBuffer.wrap(new byte[] {17}), ExternalFbinReference.HEADER_BYTES + 3);
        }
        IOException failure = expectThrows(IOException.class, reader::checkIntegrity);
        assertTrue(failure.getMessage().contains("SHA-256 mismatch"));
      }
    }
  }

  public void testCorruptDescriptorChecksumIsRejected() throws Exception {
    float[][] vectors = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Path fbin = writeFbin(createTempDir().resolve("vectors.fbin"), vectors);

    try (Directory indexDirectory = newDirectory();
        ExternalFbinFileRegistry.Registration registration =
            ExternalFbinFileRegistry.register(fbin, sha256(fbin))) {
      SegmentFixture fixture =
          writeDescriptor(indexDirectory, registration.reference(0, vectors.length));
      corruptDescriptorPayload(indexDirectory, fixture.readState());

      expectThrows(
          CorruptIndexException.class,
          () -> new ExternalFbinFlatVectorsReader(fixture.readState()));
    }
  }

  public void testDimensionBoundariesWithSlicedRowsAcrossMmapChunks() throws Exception {
    for (int dimensions : GENERALIZATION_DIMENSIONS) {
      float[][] vectors = createVectors(5, dimensions);
      Path fbin = writeFbin(createTempDir().resolve("vectors-" + dimensions + "d.fbin"), vectors);

      try (Directory indexDirectory = newDirectory();
          ExternalFbinFileRegistry.Registration registration =
              ExternalFbinFileRegistry.register(fbin, sha256(fbin))) {
        ExternalFbinReference reference = registration.reference(1, 3);
        SegmentFixture fixture = writeDescriptor(indexDirectory, reference);

        try (ExternalFbinFlatVectorsReader reader =
            new ExternalFbinFlatVectorsReader(fixture.readState(), FORCED_MMAP_CHUNK_SIZE)) {
          FloatVectorValues values = reader.getFloatVectorValues("vector");
          assertEquals(3, values.size());
          assertEquals(dimensions, values.dimension());
          assertArrayEquals(vectors[1], values.vectorValue(0), 0.0f);
          assertArrayEquals(vectors[3], values.copy().vectorValue(2), 0.0f);

          RandomVectorScorer scorer = reader.getRandomVectorScorer("vector", vectors[2]);
          assertEquals(1.0f, scorer.score(1), 0.0f);
          assertTrue(scorer.score(0) < scorer.score(1));
          reader.checkIntegrity();
        }
      }
    }
  }

  public void testSparseFileReadsFinalRowAcrossTwoGiBBoundary() throws Exception {
    int dimensions = 64;
    long rowBytes = (long) dimensions * Float.BYTES;
    long twoGiB = 1L << 31;
    int fileRows = Math.toIntExact((twoGiB - ExternalFbinReference.HEADER_BYTES) / rowBytes + 1L);
    int finalRow = fileRows - 1;
    long finalRowOffset =
        ExternalFbinReference.HEADER_BYTES + Math.multiplyExact((long) finalRow, rowBytes);
    assertTrue(finalRowOffset < twoGiB);
    assertTrue(finalRowOffset + rowBytes > twoGiB);

    float[] expected = createVectors(1, dimensions)[0];
    Path fbin = createTempDir().resolve("sparse-over-2gib.fbin");
    writeSparseFbin(fbin, fileRows, dimensions, finalRowOffset, expected);
    assertTrue(Files.size(fbin) > twoGiB);

    // Offset behavior is under test here. A synthetic content ID avoids reading the 2 GiB hole.
    String syntheticContentId = "a5".repeat(ExternalFbinReference.SHA256_BYTES);
    try (Directory indexDirectory = newDirectory();
        ExternalFbinFileRegistry.Registration registration =
            ExternalFbinFileRegistry.register(fbin, syntheticContentId)) {
      SegmentFixture fixture = writeDescriptor(indexDirectory, registration.reference(finalRow, 1));

      try (ExternalFbinFlatVectorsReader reader =
          new ExternalFbinFlatVectorsReader(fixture.readState(), 1L << 30)) {
        FloatVectorValues values = reader.getFloatVectorValues("vector");
        assertEquals(1, values.size());
        assertEquals(dimensions, values.dimension());
        assertArrayEquals(expected, values.vectorValue(0), 0.0f);
      }
    }
  }

  public void testSparseHundredMillionBy2048SliceUsesLongOffsetsAcrossMmapChunks()
      throws Exception {
    int fileRows = 100_000_000;
    int dimensions = 2048;
    long rowBytes = Math.multiplyExact((long) dimensions, Float.BYTES);
    long payloadBytes = Math.multiplyExact((long) fileRows, rowBytes);
    long fileLength = Math.addExact(ExternalFbinReference.HEADER_BYTES, payloadBytes);
    long mmapChunkSize = 1L << 30;
    assertEquals(819_200_000_000L, payloadBytes);
    assertEquals(819_200_000_008L, fileLength);

    long finalChunkStart = Math.floorDiv(fileLength - 1L, mmapChunkSize) * mmapChunkSize;
    long crossedChunkBoundary = finalChunkStart - 2L * mmapChunkSize;
    int firstRow =
        Math.toIntExact(
            Math.floorDiv(crossedChunkBoundary - ExternalFbinReference.HEADER_BYTES, rowBytes));
    int rowCount = Math.toIntExact((long) fileRows - firstRow);
    int finalRow = fileRows - 1;
    long firstRowOffset =
        Math.addExact(
            ExternalFbinReference.HEADER_BYTES, Math.multiplyExact((long) firstRow, rowBytes));
    long finalRowOffset =
        Math.addExact(
            ExternalFbinReference.HEADER_BYTES, Math.multiplyExact((long) finalRow, rowBytes));
    assertTrue(firstRowOffset < crossedChunkBoundary);
    assertTrue(firstRowOffset + rowBytes > crossedChunkBoundary);
    assertTrue(Math.multiplyExact((long) rowCount, rowBytes) > Integer.MAX_VALUE);

    float[] firstExpected = createVector(dimensions, 0.25f);
    float[] finalExpected = createVector(dimensions, -0.5f);
    Path fbin = createTempDir().resolve("sparse-100m-2048d.fbin");
    writeSparseFbinRows(
        fbin,
        fileRows,
        dimensions,
        new SparseRow(firstRowOffset, firstExpected),
        new SparseRow(finalRowOffset, finalExpected));
    assertEquals(fileLength, Files.size(fbin));

    // The synthetic identity deliberately avoids hashing or otherwise reading the sparse hole.
    String syntheticContentId = "c7".repeat(ExternalFbinReference.SHA256_BYTES);
    try (Directory indexDirectory = newDirectory();
        ExternalFbinFileRegistry.Registration registration =
            ExternalFbinFileRegistry.register(fbin, syntheticContentId)) {
      ExternalFbinReference reference = registration.reference(firstRow, rowCount);
      assertEquals(fileLength, reference.fileLength());
      assertEquals(firstRow, reference.firstRow());
      assertEquals(rowCount, reference.rows());
      assertEquals(dimensions, reference.dimensions());
      assertEquals(firstRowOffset, reference.payloadOffset());
      assertEquals(Math.multiplyExact((long) rowCount, rowBytes), reference.payloadLength());
      SegmentFixture fixture = writeDescriptor(indexDirectory, reference);

      try (ExternalFbinFlatVectorsReader reader =
          new ExternalFbinFlatVectorsReader(fixture.readState(), mmapChunkSize)) {
        FloatVectorValues values = reader.getFloatVectorValues("vector");
        assertEquals(rowCount, values.size());
        assertEquals(dimensions, values.dimension());
        assertArrayEquals(firstExpected, values.vectorValue(0), 0.0f);

        int finalOrdinal = rowCount - 1;
        long finalComponentOffset =
            Math.addExact(
                Math.multiplyExact((long) finalOrdinal, rowBytes),
                Math.multiplyExact((long) dimensions - 1L, Float.BYTES));
        assertTrue(finalComponentOffset > Integer.MAX_VALUE);
        float[] finalActual = values.copy().vectorValue(finalOrdinal);
        assertEquals(finalExpected[0], finalActual[0], 0.0f);
        assertEquals(finalExpected[dimensions - 1], finalActual[dimensions - 1], 0.0f);
        assertArrayEquals(finalExpected, finalActual, 0.0f);
      }
    }
  }

  public void testReaderRequiresRegisteredContentId() throws Exception {
    float[][] vectors = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Path fbin = writeFbin(createTempDir().resolve("vectors.fbin"), vectors);
    String digest = sha256(fbin);

    try (Directory indexDirectory = newDirectory()) {
      SegmentFixture fixture;
      try (ExternalFbinFileRegistry.Registration registration =
          ExternalFbinFileRegistry.register(fbin, digest)) {
        fixture = writeDescriptor(indexDirectory, registration.reference(0, vectors.length));
      }

      IOException failure =
          expectThrows(
              IOException.class, () -> new ExternalFbinFlatVectorsReader(fixture.readState()));
      assertTrue(failure.getMessage().contains("No allowlisted external FBIN"));
    }
  }

  public void testIntegrityReportsOverflowingMutableHeaderAsCorruption() throws Exception {
    float[][] vectors = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Path fbin = writeFbin(createTempDir().resolve("vectors.fbin"), vectors);
    String digest = sha256(fbin);

    try (Directory indexDirectory = newDirectory();
        ExternalFbinFileRegistry.Registration registration =
            ExternalFbinFileRegistry.register(fbin, digest)) {
      SegmentFixture fixture =
          writeDescriptor(indexDirectory, registration.reference(0, vectors.length));
      try (ExternalFbinFlatVectorsReader reader =
          new ExternalFbinFlatVectorsReader(fixture.readState())) {
        ByteBuffer corruptHeader =
            ByteBuffer.allocate((int) ExternalFbinReference.HEADER_BYTES)
                .order(ByteOrder.LITTLE_ENDIAN)
                .putInt(Integer.MAX_VALUE)
                .putInt(Integer.MAX_VALUE);
        corruptHeader.flip();
        try (FileChannel channel = FileChannel.open(fbin, StandardOpenOption.WRITE)) {
          while (corruptHeader.hasRemaining()) {
            channel.write(corruptHeader, corruptHeader.position());
          }
        }
        expectThrows(CorruptIndexException.class, reader::checkIntegrity);
      }
    }
  }

  public void testExternalMarkerIsScopedToSegmentSuffix() throws Exception {
    float[][] vectors = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Path fbin = writeFbin(createTempDir().resolve("vectors.fbin"), vectors);
    String digest = sha256(fbin);

    try (Directory indexDirectory = newDirectory();
        ExternalFbinFileRegistry.Registration registration =
            ExternalFbinFileRegistry.register(fbin, digest)) {
      SegmentFixture fixture =
          writeDescriptor(indexDirectory, registration.reference(0, vectors.length));
      SegmentReadState base = fixture.readState();
      base.segmentInfo.putAttribute(
          ExternalFbinReferenceWriter.segmentAttribute("external"),
          ExternalFbinReferenceWriter.SEGMENT_ATTRIBUTE_VALUE);

      assertTrue(
          ExternalFbinFlatVectorsReader.hasExternalMarker(new SegmentReadState(base, "external")));
      assertFalse(
          ExternalFbinFlatVectorsReader.hasExternalMarker(
              new SegmentReadState(base, "conventional")));
    }
  }

  public void testOrphanDescriptorDoesNotSelectExternalReader() throws Exception {
    float[][] vectors = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Path fbin = writeFbin(createTempDir().resolve("vectors.fbin"), vectors);

    try (Directory indexDirectory = newDirectory();
        ExternalFbinFileRegistry.Registration registration =
            ExternalFbinFileRegistry.register(fbin, sha256(fbin))) {
      SegmentFixture fixture =
          writeDescriptor(indexDirectory, registration.reference(0, vectors.length));
      SegmentReadState state = fixture.readState();
      state.segmentInfo.putAttribute(ExternalFbinReferenceWriter.segmentAttribute(""), null);
      assertFalse(ExternalFbinFlatVectorsReader.hasExternalMarker(state));

      IOException failure =
          expectThrows(
              IOException.class,
              () -> new Lucene99AcceleratedHNSWVectorsFormat().fieldsReader(state));
      assertFalse(
          "An uncommitted descriptor must not override the committed segment marker: " + failure,
          failure.getMessage().contains("marker/descriptor mismatch"));
    }
  }

  private static SegmentFixture writeDescriptor(
      Directory directory, ExternalFbinReference reference) throws Exception {
    FieldInfo fieldInfo =
        new FieldInfo(
            "vector",
            0,
            false,
            false,
            false,
            IndexOptions.NONE,
            DocValuesType.NONE,
            DocValuesSkipIndexType.NONE,
            -1L,
            Map.of(),
            0,
            0,
            0,
            reference.dimensions(),
            VectorEncoding.FLOAT32,
            EUCLIDEAN,
            false,
            false);
    FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] {fieldInfo});
    SegmentInfo segmentInfo =
        new SegmentInfo(
            directory,
            Version.LATEST,
            Version.LATEST,
            "_0",
            reference.rows(),
            false,
            false,
            Codec.getDefault(),
            Map.of(),
            StringHelper.randomId(),
            Map.of(),
            null);
    segmentInfo.putAttribute(
        ExternalFbinReferenceWriter.segmentAttribute(""),
        ExternalFbinReferenceWriter.SEGMENT_ATTRIBUTE_VALUE);
    SegmentWriteState writeState =
        new SegmentWriteState(
            InfoStream.NO_OUTPUT, directory, segmentInfo, fieldInfos, null, IOContext.DEFAULT);
    try (ExternalFbinReferenceWriter writer = new ExternalFbinReferenceWriter(writeState)) {
      writer.writeField(fieldInfo, reference);
      writer.finish();
    }
    return new SegmentFixture(
        new SegmentReadState(directory, segmentInfo, fieldInfos, IOContext.DEFAULT));
  }

  private static void corruptDescriptorPayload(Directory directory, SegmentReadState state)
      throws IOException {
    String fileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, ExternalFbinReferenceWriter.EXTENSION);
    byte[] bytes;
    try (IndexInput input = directory.openInput(fileName, IOContext.DEFAULT)) {
      bytes = new byte[Math.toIntExact(input.length())];
      input.readBytes(bytes, 0, bytes.length);
    }
    int lastDigestByte = bytes.length - CodecUtil.footerLength() - Integer.BYTES - 1;
    bytes[lastDigestByte] ^= 1;
    String replacement;
    try (IndexOutput output =
        directory.createTempOutput(
            "corrupt", ExternalFbinReferenceWriter.EXTENSION, IOContext.DEFAULT)) {
      replacement = output.getName();
      output.writeBytes(bytes, bytes.length);
    }
    directory.deleteFile(fileName);
    directory.rename(replacement, fileName);
  }

  private static Path writeFbin(Path file, float[][] vectors) throws IOException {
    int rows = vectors.length;
    int dimensions = vectors[0].length;
    ByteBuffer data =
        ByteBuffer.allocate(2 * Integer.BYTES + rows * dimensions * Float.BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
    data.putInt(rows);
    data.putInt(dimensions);
    for (float[] vector : vectors) {
      for (float value : vector) {
        data.putFloat(value);
      }
    }
    data.flip();
    try (FileChannel channel =
        FileChannel.open(file, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
      while (data.hasRemaining()) {
        channel.write(data);
      }
    }
    return file;
  }

  private static void writeSparseFbin(
      Path file, int rows, int dimensions, long finalRowOffset, float[] finalRow)
      throws IOException {
    ByteBuffer header =
        ByteBuffer.allocate((int) ExternalFbinReference.HEADER_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN)
            .putInt(rows)
            .putInt(dimensions);
    header.flip();
    ByteBuffer row = ByteBuffer.allocate(dimensions * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (float value : finalRow) {
      row.putFloat(value);
    }
    row.flip();

    try (FileChannel channel =
        FileChannel.open(file, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
      writeFully(channel, header, 0L);
      writeFully(channel, row, finalRowOffset);
    }
  }

  private static void writeSparseFbinRows(
      Path file, int rows, int dimensions, SparseRow... sparseRows) throws IOException {
    ByteBuffer header =
        ByteBuffer.allocate((int) ExternalFbinReference.HEADER_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN)
            .putInt(rows)
            .putInt(dimensions);
    header.flip();

    try (FileChannel channel =
        FileChannel.open(file, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
      writeFully(channel, header, 0L);
      for (SparseRow sparseRow : sparseRows) {
        assertEquals(dimensions, sparseRow.values().length);
        ByteBuffer row =
            ByteBuffer.allocate(dimensions * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (float value : sparseRow.values()) {
          row.putFloat(value);
        }
        row.flip();
        writeFully(channel, row, sparseRow.offset());
      }
    }
  }

  private static void writeFully(FileChannel channel, ByteBuffer source, long position)
      throws IOException {
    while (source.hasRemaining()) {
      int written = channel.write(source, position + source.position());
      if (written == 0) {
        throw new IOException("Unable to make progress writing test FBIN");
      }
    }
  }

  private static float[][] createVectors(int rows, int dimensions) {
    float[][] vectors = new float[rows][dimensions];
    for (int row = 0; row < rows; row++) {
      for (int dimension = 0; dimension < dimensions; dimension++) {
        vectors[row][dimension] = row * 0.25f + dimension * 0.03125f;
      }
    }
    return vectors;
  }

  private static float[] createVector(int dimensions, float base) {
    float[] vector = new float[dimensions];
    for (int dimension = 0; dimension < dimensions; dimension++) {
      vector[dimension] = base + dimension * 0.03125f;
    }
    return vector;
  }

  private static String sha256(Path file) throws Exception {
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    digest.update(Files.readAllBytes(file));
    return HexFormat.of().formatHex(digest.digest());
  }

  private record SegmentFixture(SegmentReadState readState) {}

  private record SparseRow(long offset, float[] values) {}
}

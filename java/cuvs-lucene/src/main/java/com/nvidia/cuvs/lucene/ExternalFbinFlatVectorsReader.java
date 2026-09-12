/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.lucene95.OffHeapFloatVectorValues.DenseOffHeapVectorValues;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.store.ReadAdvice;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

/** Reads dense float32 vectors from a registered immutable FBIN instead of a Lucene {@code .vec}. */
final class ExternalFbinFlatVectorsReader extends FlatVectorsReader {

  private static final long SHALLOW_SIZE =
      RamUsageEstimator.shallowSizeOfInstance(ExternalFbinFlatVectorsReader.class);

  private final FieldInfo fieldInfo;
  private final ExternalFbinReference reference;
  private final ExternalFbinMappedInputCache.Lease sourceLease;
  private final IndexInput payloadInput;

  ExternalFbinFlatVectorsReader(SegmentReadState state) throws IOException {
    this(state, MMapDirectory.DEFAULT_MAX_CHUNK_SIZE);
  }

  ExternalFbinFlatVectorsReader(SegmentReadState state, long maxMmapChunkSize) throws IOException {
    super(DefaultFlatVectorScorer.INSTANCE);
    Descriptor descriptor = readDescriptor(state);
    fieldInfo = descriptor.fieldInfo();
    reference = descriptor.reference();

    var sourcePath = ExternalFbinIO.validateAndResolve(reference);
    ExternalFbinMappedInputCache.Lease newSourceLease = null;
    IndexInput newPayload = null;
    boolean success = false;
    try {
      IOContext context = state.context.withReadAdvice(ReadAdvice.RANDOM);
      newSourceLease =
          ExternalFbinMappedInputCache.acquire(sourcePath, reference, maxMmapChunkSize, context);
      IndexInput newSource = newSourceLease.sourceInput();
      validateOpenedSource(newSource, reference);
      newPayload =
          newSource.slice(
              "external-fbin-payload",
              reference.payloadOffset(),
              reference.payloadLength(),
              ReadAdvice.RANDOM);
      success = true;
    } finally {
      if (!success) {
        IOUtils.closeWhileHandlingException(newPayload, newSourceLease);
      }
    }
    sourceLease = newSourceLease;
    payloadInput = newPayload;
  }

  static boolean hasExternalMarker(SegmentReadState state) throws IOException {
    String segmentMarker =
        state.segmentInfo.getAttribute(
            ExternalFbinReferenceWriter.segmentAttribute(state.segmentSuffix));
    if (segmentMarker != null
        && !ExternalFbinReferenceWriter.SEGMENT_ATTRIBUTE_VALUE.equals(segmentMarker)) {
      throw new CorruptIndexException(
          "Unsupported external FBIN segment marker version " + segmentMarker,
          state.segmentInfo.name);
    }
    return segmentMarker != null;
  }

  private static Descriptor readDescriptor(SegmentReadState state) throws IOException {
    String fileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, ExternalFbinReferenceWriter.EXTENSION);
    try (ChecksumIndexInput input = state.directory.openChecksumInput(fileName)) {
      Throwable prior = null;
      try {
        CodecUtil.checkIndexHeader(
            input,
            ExternalFbinReferenceWriter.CODEC_NAME,
            ExternalFbinReferenceWriter.VERSION_START,
            ExternalFbinReferenceWriter.VERSION_CURRENT,
            state.segmentInfo.getId(),
            state.segmentSuffix);
        int fieldNumber = input.readInt();
        if (fieldNumber < 0) {
          throw new CorruptIndexException("External FBIN descriptor contains no field", input);
        }
        FieldInfo field = state.fieldInfos.fieldInfo(fieldNumber);
        if (field == null) {
          throw new CorruptIndexException(
              "External FBIN descriptor has unknown field number " + fieldNumber, input);
        }
        String fieldName = input.readString();
        String encodingName = input.readString();
        String similarityName = input.readString();
        int dimensions = input.readVInt();
        int rows = input.readInt();
        int fileRows = input.readInt();
        long firstRow = input.readLong();
        long fileLength = input.readLong();
        long payloadOffset = input.readLong();
        long payloadLength = input.readLong();
        byte[] sha256 = new byte[ExternalFbinReference.SHA256_BYTES];
        input.readBytes(sha256, 0, sha256.length);
        if (input.readInt() != -1) {
          throw new CorruptIndexException(
              "External FBIN descriptor contains more than one field", input);
        }

        final VectorEncoding encoding;
        final VectorSimilarityFunction similarity;
        try {
          encoding = VectorEncoding.valueOf(encodingName);
          similarity = VectorSimilarityFunction.valueOf(similarityName);
        } catch (IllegalArgumentException e) {
          throw new CorruptIndexException(
              "Invalid vector metadata in external FBIN descriptor", input, e);
        }
        if (!field.name.equals(fieldName)
            || encoding != VectorEncoding.FLOAT32
            || field.getVectorEncoding() != encoding
            || field.getVectorSimilarityFunction() != similarity
            || field.getVectorDimension() != dimensions) {
          throw new CorruptIndexException(
              "External FBIN descriptor does not match FieldInfo for " + field.name, input);
        }
        if (rows != state.segmentInfo.maxDoc()) {
          throw new CorruptIndexException(
              "External FBIN rows "
                  + rows
                  + " do not match dense segment maxDoc "
                  + state.segmentInfo.maxDoc(),
              input);
        }

        ExternalFbinReference reference;
        try {
          reference =
              ExternalFbinReference.fromDescriptor(
                  sha256, fileLength, payloadOffset, payloadLength, rows, dimensions);
        } catch (ArithmeticException | IllegalArgumentException e) {
          throw new CorruptIndexException("Invalid external FBIN range metadata", input, e);
        }
        long rowBytes = Math.multiplyExact((long) dimensions, Float.BYTES);
        long derivedFileRows =
            (reference.fileLength() - ExternalFbinReference.HEADER_BYTES) / rowBytes;
        if (fileRows <= 0
            || fileRows != derivedFileRows
            || firstRow != reference.firstRow()
            || firstRow + rows > fileRows) {
          throw new CorruptIndexException("Inconsistent external FBIN row metadata", input);
        }
        return new Descriptor(field, reference);
      } catch (Throwable t) {
        prior = t;
        throw t;
      } finally {
        CodecUtil.checkFooter(input, prior);
      }
    }
  }

  private static void validateOpenedSource(IndexInput source, ExternalFbinReference reference)
      throws IOException {
    if (source.length() != reference.fileLength()) {
      throw new CorruptIndexException(
          "External FBIN length changed: expected "
              + reference.fileLength()
              + " but got "
              + source.length(),
          source);
    }
    byte[] headerBytes = new byte[(int) ExternalFbinReference.HEADER_BYTES];
    source.seek(0L);
    source.readBytes(headerBytes, 0, headerBytes.length);
    ByteBuffer header = ByteBuffer.wrap(headerBytes).order(ByteOrder.LITTLE_ENDIAN);
    int fileRows = header.getInt();
    int dimensions = header.getInt();
    if (fileRows <= 0 || dimensions <= 0) {
      throw new CorruptIndexException(
          "External FBIN header contains a non-positive shape: " + fileRows + " x " + dimensions,
          source);
    }
    final long expectedLength;
    try {
      long rowBytes = Math.multiplyExact((long) dimensions, Float.BYTES);
      expectedLength =
          Math.addExact(
              ExternalFbinReference.HEADER_BYTES, Math.multiplyExact((long) fileRows, rowBytes));
    } catch (ArithmeticException e) {
      throw new CorruptIndexException(
          "External FBIN header shape overflows its file length", source, e);
    }
    if (dimensions != reference.dimensions()
        || expectedLength != source.length()
        || reference.firstRow() + reference.rows() > fileRows) {
      throw new CorruptIndexException(
          "External FBIN header or shape does not match the segment descriptor", source);
    }
  }

  private void requireField(String field, VectorEncoding expectedEncoding) {
    if (!fieldInfo.name.equals(field)) {
      throw new IllegalArgumentException("field=\"" + field + "\" not found");
    }
    if (fieldInfo.getVectorEncoding() != expectedEncoding) {
      throw new IllegalArgumentException(
          "field=\""
              + field
              + "\" is encoded as "
              + fieldInfo.getVectorEncoding()
              + ", expected "
              + expectedEncoding);
    }
  }

  @Override
  public FloatVectorValues getFloatVectorValues(String field) throws IOException {
    requireField(field, VectorEncoding.FLOAT32);
    return new DenseOffHeapVectorValues(
        reference.dimensions(),
        reference.rows(),
        payloadInput.clone(),
        Math.multiplyExact(reference.dimensions(), Float.BYTES),
        vectorScorer,
        fieldInfo.getVectorSimilarityFunction());
  }

  @Override
  public ByteVectorValues getByteVectorValues(String field) {
    requireField(field, VectorEncoding.BYTE);
    throw new AssertionError("External FBIN fields cannot use byte encoding");
  }

  @Override
  public RandomVectorScorer getRandomVectorScorer(String field, float[] target) throws IOException {
    requireField(field, VectorEncoding.FLOAT32);
    return vectorScorer.getRandomVectorScorer(
        fieldInfo.getVectorSimilarityFunction(), getFloatVectorValues(field), target);
  }

  @Override
  public RandomVectorScorer getRandomVectorScorer(String field, byte[] target) {
    requireField(field, VectorEncoding.BYTE);
    throw new AssertionError("External FBIN fields cannot use byte encoding");
  }

  @Override
  public void checkIntegrity() throws IOException {
    IndexInput sourceInput = sourceLease.sourceInput();
    IndexInput clone = sourceInput.clone();
    // Lucene IndexInput clones are non-owning. Multi-chunk MemorySegmentIndexInput clones share
    // the owner's segment array, so closing a clone would invalidate the live reader.
    validateOpenedSource(clone, reference);
    ExternalFbinIO.verifySha256(sourceInput, reference);
  }

  @Override
  public FlatVectorsReader getMergeInstance() {
    try {
      payloadInput.updateReadAdvice(ReadAdvice.SEQUENTIAL);
    } catch (IOException e) {
      throw new IllegalStateException("Unable to prepare external FBIN reader for merge", e);
    }
    return this;
  }

  @Override
  public void finishMerge() throws IOException {
    payloadInput.updateReadAdvice(ReadAdvice.RANDOM);
  }

  @Override
  public long ramBytesUsed() {
    return SHALLOW_SIZE;
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(payloadInput, sourceLease);
  }

  private record Descriptor(FieldInfo fieldInfo, ExternalFbinReference reference) {}
}

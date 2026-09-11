/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.io.Closeable;
import java.io.IOException;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;

final class ExternalFbinReferenceWriter implements Closeable {

  static final String EXTENSION = "vefr";
  static final String CODEC_NAME = "CuVSExternalFbinReference";
  static final int VERSION_START = 0;
  static final int VERSION_CURRENT = 0;
  static final String SEGMENT_ATTRIBUTE_PREFIX = "cuvs.external_fbin.";
  static final String SEGMENT_ATTRIBUTE_VALUE = "1";

  private final IndexOutput output;
  private boolean wroteField;
  private boolean finished;

  ExternalFbinReferenceWriter(SegmentWriteState state) throws IOException {
    String previous =
        state.segmentInfo.putAttribute(
            segmentAttribute(state.segmentSuffix), SEGMENT_ATTRIBUTE_VALUE);
    if (previous != null && !SEGMENT_ATTRIBUTE_VALUE.equals(previous)) {
      throw new IllegalStateException(
          "Segment already has an incompatible external FBIN marker: " + previous);
    }
    String fileName = fileName(state);
    IndexOutput newOutput = null;
    boolean success = false;
    try {
      newOutput = state.directory.createOutput(fileName, state.context);
      CodecUtil.writeIndexHeader(
          newOutput, CODEC_NAME, VERSION_CURRENT, state.segmentInfo.getId(), state.segmentSuffix);
      success = true;
    } finally {
      if (!success) {
        IOUtils.closeWhileHandlingException(newOutput);
      }
    }
    output = newOutput;
  }

  static String fileName(SegmentWriteState state) {
    return IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, EXTENSION);
  }

  static String segmentAttribute(String segmentSuffix) {
    return SEGMENT_ATTRIBUTE_PREFIX + segmentSuffix;
  }

  void writeField(FieldInfo field, ExternalFbinReference reference) throws IOException {
    if (finished) {
      throw new IllegalStateException("External FBIN reference writer is already finished");
    }
    if (wroteField) {
      throw new UnsupportedOperationException(
          "Immutable external FBIN mode supports exactly one vector field per segment");
    }
    if (field.getVectorEncoding() != VectorEncoding.FLOAT32) {
      throw new IllegalArgumentException("External FBIN references require FLOAT32 vectors");
    }
    if (field.getVectorDimension() != reference.dimensions()) {
      throw new IllegalArgumentException(
          "Field dimension does not match the external FBIN reference");
    }

    long rowBytes = Math.multiplyExact((long) reference.dimensions(), Float.BYTES);
    long fullPayloadBytes = reference.fileLength() - ExternalFbinReference.HEADER_BYTES;
    if (Math.floorMod(fullPayloadBytes, rowBytes) != 0) {
      throw new IllegalArgumentException("FBIN file length is not an exact number of rows");
    }
    int fileRows = Math.toIntExact(fullPayloadBytes / rowBytes);

    output.writeInt(field.number);
    output.writeString(field.name);
    output.writeString(field.getVectorEncoding().name());
    output.writeString(field.getVectorSimilarityFunction().name());
    output.writeVInt(reference.dimensions());
    output.writeInt(reference.rows());
    output.writeInt(fileRows);
    output.writeLong(reference.firstRow());
    output.writeLong(reference.fileLength());
    output.writeLong(reference.payloadOffset());
    output.writeLong(reference.payloadLength());
    output.writeBytes(reference.sha256(), ExternalFbinReference.SHA256_BYTES);
    wroteField = true;
  }

  void finish() throws IOException {
    if (finished) {
      throw new IllegalStateException("External FBIN reference writer is already finished");
    }
    if (!wroteField) {
      throw new IllegalStateException("No external FBIN field was written");
    }
    finished = true;
    output.writeInt(-1);
    CodecUtil.writeFooter(output);
  }

  @Override
  public void close() throws IOException {
    output.close();
  }
}

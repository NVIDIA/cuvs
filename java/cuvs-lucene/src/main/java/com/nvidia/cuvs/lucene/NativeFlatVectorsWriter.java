/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CuVSHostMatrix;
import java.io.Closeable;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;

/**
 * Writes the flat vector files ({@code .vec} data + {@code .vemf} meta) directly from a native host
 * matrix, byte-for-byte compatible with Lucene's {@code Lucene99FlatVectorsWriter} so the stock
 * {@code Lucene99FlatVectorsReader} can read them.
 *
 * <p>This is the hint-path counterpart to delegating to Lucene's {@code FlatVectorsWriter}: the
 * accelerated writer streams vectors into a {@link CuVSHostMatrix} during indexing (see
 * {@link FieldWriter}) and never materialises the full dataset as a {@code List<float[]>} on the
 * Java heap, so the flat file is written here from that native matrix instead.
 *
 * <p><b>Ported code — version-pinned.</b> The dense float32 layout (format constants, header, meta
 * field order, and footer) is transcribed from Lucene <b>10.2.0</b>, which must stay equal to the
 * {@code lucene-core} version in {@code pom.xml}. This is enforced by
 * {@code TestNativeFlatVectorsWriterFormatConstants}, which fails the build if the resolved
 * {@code lucene-core} classpath version drifts from the pin below — update
 * {@code PINNED_LUCENE_VERSION} there together with this class on a verified upgrade. Sources (tag
 * {@code releases/lucene/10.2.0}):
 *
 * <ul>
 *   <li>{@code org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat} — format constants:
 *       https://github.com/apache/lucene/blob/releases/lucene/10.2.0/lucene/core/src/java/org/apache/lucene/codecs/lucene99/Lucene99FlatVectorsFormat.java
 *   <li>{@code org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsWriter} — write sequence:
 *       https://github.com/apache/lucene/blob/releases/lucene/10.2.0/lucene/core/src/java/org/apache/lucene/codecs/lucene99/Lucene99FlatVectorsWriter.java
 * </ul>
 *
 * <p><b>On a Lucene upgrade:</b> re-verify this class against the new version's two files above. The
 * constants are package-private in Lucene (hence re-declared here), and the file format is validated
 * on read via {@code CodecUtil.checkIndexHeader} (codec name + version range) plus the fixed meta
 * layout — so if the new version bumps {@code VERSION_CURRENT}, renames a codec, or changes the meta
 * field order, files written here will be silently incompatible and the stock
 * {@code Lucene99FlatVectorsReader} will reject or misread them. Concretely:
 *
 * <ol>
 *   <li>Diff the new version's {@code Lucene99FlatVectorsFormat}/{@code Lucene99FlatVectorsWriter}
 *       sources against the ones linked above and mirror any changed constants and any change to
 *       the {@code writeField}/{@code writeMeta} byte sequence (header, meta field order, footer)
 *       here. This class writes directly from native memory to avoid the per-vector {@code
 *       FloatVectorValues} indirection Lucene's own writer requires — that's the reason to keep
 *       hand-porting the format rather than delegating to it.
 *   <li>Bump {@code PINNED_LUCENE_VERSION} in {@code TestNativeFlatVectorsWriterFormatConstants} to
 *       clear the tripwire.
 *   <li>Note: {@code TestNativeFlatVectorsWriterRoundTrip} positively verifies the result
 *       by building a small index with {@code numInputVectors} set and asserting every
 *       vector round-trips byte-exact through the stock {@code Lucene99FlatVectorsReader}.
 * </ol>
 */
final class NativeFlatVectorsWriter implements Closeable {

  // Mirrors org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat (10.2.0) so the standard
  // Lucene99FlatVectorsReader accepts the header/codec of the files written here.
  private static final String META_CODEC_NAME = "Lucene99FlatVectorsFormatMeta";
  private static final String VECTOR_DATA_CODEC_NAME = "Lucene99FlatVectorsFormatData";
  private static final String META_EXTENSION = "vemf";
  private static final String VECTOR_DATA_EXTENSION = "vec";
  private static final int VERSION_CURRENT = 0;
  private static final int DIRECT_MONOTONIC_BLOCK_SHIFT = 16;

  // Little-endian float layout matching Lucene's on-disk .vec byte order. Must be UNALIGNED: the
  // destination is a heap byte[]-backed MemorySegment whose max alignment is 1 byte, so a 4-byte
  // aligned JAVA_FLOAT layout is rejected with "incompatible with alignment constraints".
  private static final ValueLayout.OfFloat LE_FLOAT =
      ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  // Byte granularity for a single writeBytes call; bounds the transient encode buffer.
  private static final int CHUNK_BYTES = 1 << 18; // 256 KiB

  private final IndexOutput meta;
  private final IndexOutput vectorData;
  private boolean finished;

  NativeFlatVectorsWriter(SegmentWriteState state) throws IOException {
    String metaFileName =
        IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, META_EXTENSION);
    String vectorDataFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, VECTOR_DATA_EXTENSION);
    boolean success = false;
    try {
      meta = state.directory.createOutput(metaFileName, state.context);
      vectorData = state.directory.createOutput(vectorDataFileName, state.context);
      CodecUtil.writeIndexHeader(
          meta, META_CODEC_NAME, VERSION_CURRENT, state.segmentInfo.getId(), state.segmentSuffix);
      CodecUtil.writeIndexHeader(
          vectorData,
          VECTOR_DATA_CODEC_NAME,
          VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      success = true;
    } finally {
      if (success == false) {
        IOUtils.closeWhileHandlingException(this);
      }
    }
  }

  /**
   * Writes one dense float32 field: the raw vectors to {@code .vec} and the field metadata (plus the
   * ordinal-to-doc mapping) to {@code .vemf}. Vectors are read from {@code matrix} in ordinal order,
   * which matches the ascending-docID order in which {@code docsWithField} was populated.
   *
   * @param field the field being written
   * @param matrix the native host matrix holding {@code docsWithField.cardinality()} rows of {@code
   *     field.getVectorDimension()} floats each
   * @param maxDoc the segment's maxDoc, used to build the ordinal-to-doc mapping
   * @param docsWithField the set of docs that have a value for this field
   */
  void writeField(
      FieldInfo field, CuVSHostMatrix matrix, int maxDoc, DocsWithFieldSet docsWithField)
      throws IOException {
    // Mirrors Lucene99FlatVectorsWriter#writeField (see class-level version pin).
    int count = docsWithField.cardinality();
    int dim = field.getVectorDimension();
    long vectorDataOffset = vectorData.alignFilePointer(Float.BYTES);
    writeFloat32Vectors(matrix, count, dim);
    long vectorDataLength = vectorData.getFilePointer() - vectorDataOffset;
    writeMeta(field, maxDoc, count, vectorDataOffset, vectorDataLength, docsWithField);
  }

  private void writeFloat32Vectors(CuVSHostMatrix matrix, int count, int dim) throws IOException {
    int rowBytes = dim * Float.BYTES;
    int chunkRows = Math.max(1, CHUNK_BYTES / rowBytes);
    byte[] chunk = new byte[chunkRows * rowBytes];
    MemorySegment chunkSeg = MemorySegment.ofArray(chunk);
    float[] rowBuf = new float[dim];
    int r = 0;
    for (int ord = 0; ord < count; ord++) {
      matrix.getRow(ord).toArray(rowBuf); // native -> heap float[] (bulk)
      MemorySegment.copy(rowBuf, 0, chunkSeg, LE_FLOAT, (long) r * rowBytes, dim); // -> LE bytes
      if (++r == chunkRows) {
        vectorData.writeBytes(chunk, r * rowBytes);
        r = 0;
      }
    }
    if (r > 0) {
      vectorData.writeBytes(chunk, r * rowBytes);
    }
  }

  private void writeMeta(
      FieldInfo field,
      int maxDoc,
      int count,
      long vectorDataOffset,
      long vectorDataLength,
      DocsWithFieldSet docsWithField)
      throws IOException {
    // Mirrors Lucene99FlatVectorsWriter#writeMeta (see class-level version pin); field order is
    // load-bearing and must match Lucene's reader.
    meta.writeInt(field.number);
    meta.writeInt(field.getVectorEncoding().ordinal());
    meta.writeInt(field.getVectorSimilarityFunction().ordinal());
    meta.writeVLong(vectorDataOffset);
    meta.writeVLong(vectorDataLength);
    meta.writeVInt(field.getVectorDimension());
    meta.writeInt(count);
    OrdToDocDISIReaderConfiguration.writeStoredMeta(
        DIRECT_MONOTONIC_BLOCK_SHIFT, meta, vectorData, count, maxDoc, docsWithField);
  }

  /** Writes the end-of-fields marker and footers. Mirrors {@code Lucene99FlatVectorsWriter.finish}. */
  void finish() throws IOException {
    if (finished) {
      throw new IllegalStateException("already finished");
    }
    finished = true;
    if (meta != null) {
      meta.writeInt(-1);
      CodecUtil.writeFooter(meta);
    }
    if (vectorData != null) {
      CodecUtil.writeFooter(vectorData);
    }
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(meta, vectorData);
  }
}

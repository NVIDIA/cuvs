/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.AcceleratedHNSWUtils.printInfoStream;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.closeCuVSResourcesInstance;
import static org.apache.lucene.index.VectorEncoding.FLOAT32;
import static org.apache.lucene.util.RamUsageEstimator.shallowSizeOfInstance;

import com.nvidia.cuvs.CuVSHostMatrix;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter.DocMap;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.InfoStream;

/**
 * Native-flat-buffered {@link KnnVectorsWriter}, streaming vectors directly into a native host
 * matrix (see {@link NativeFieldWriter}) rather than a heap {@code List<float[]>}, and writing the
 * flat {@code .vec}/{@code .vemf} files itself via {@link NativeFlatVectorsWriter} instead of
 * delegating to Lucene's {@link org.apache.lucene.codecs.hnsw.FlatVectorsWriter}.
 *
 * <p>This writer supports only the unsorted, single-segment flush path: merges and index-sorted
 * flushes are rejected. It is package-private and constructed only by {@link
 * Lucene99AcceleratedHNSWVectorsFormat#fieldsWriter} when {@code numInputVectors > 0}, which is
 * reachable only via the package-private constructors on {@link Lucene99AcceleratedHNSWVectorsFormat}
 * and {@link Lucene101AcceleratedHNSWCodec} used exclusively by {@link CagraHnswBulkIndexWriter} —
 * the only caller that owns its {@link org.apache.lucene.index.IndexWriter} and can guarantee
 * those invariants.
 */
final class NativeFlatBufferedHNSWVectorsWriter extends KnnVectorsWriter {

  private static final long SHALLOW_RAM_BYTES_USED =
      shallowSizeOfInstance(NativeFlatBufferedHNSWVectorsWriter.class);
  private static final String COMPONENT = "NativeFlatBufferedHNSWVectorsWriter";

  private final int numInputVectors;
  private final List<NativeFieldWriter> fields = new ArrayList<>();
  private final InfoStream infoStream;
  private NativeFlatVectorsWriter nativeFlat;
  private AcceleratedHnswGraphOutput graphOutput;
  private boolean finished;

  NativeFlatBufferedHNSWVectorsWriter(
      SegmentWriteState state, AcceleratedHNSWParams acceleratedHNSWParams, int numInputVectors)
      throws IOException {
    super();
    if (numInputVectors <= 0) {
      throw new IllegalArgumentException("numInputVectors must be > 0, got " + numInputVectors);
    }
    if (state.segmentInfo.getIndexSort() != null) {
      throw new IllegalArgumentException(
          "AcceleratedHNSWParams.numInputVectors (native flat buffering) does not support"
              + " index-sorted segments; unset it (0) to use the heap-buffered path");
    }
    this.infoStream = state.infoStream;
    this.numInputVectors = numInputVectors;
    boolean success = false;
    try {
      // In hint mode we own the flat files; the Lucene flat writer must be absent to avoid opening
      // the same .vec/.vemf outputs.
      nativeFlat = new NativeFlatVectorsWriter(state);
      graphOutput = new AcceleratedHnswGraphOutput(state, acceleratedHNSWParams);
      success = true;
      printInfoStream(infoStream, COMPONENT, "NativeFlatBufferedHNSWVectorsWriter is initialized");
    } finally {
      if (success == false) {
        IOUtils.closeWhileHandlingException(this);
      }
    }
  }

  /**
   * Add new field for indexing.
   */
  @Override
  public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
    var encoding = fieldInfo.getVectorEncoding();
    if (encoding != FLOAT32) {
      throw new IllegalArgumentException("Expected float32, got:" + encoding);
    }
    // Buffer directly into a native host matrix; return the field writer itself so Lucene routes
    // addValue() here rather than to a (nonexistent) Lucene flat field writer.
    var cuvsFieldWriter = new NativeFieldWriter(fieldInfo, numInputVectors);
    fields.add(cuvsFieldWriter);
    return cuvsFieldWriter;
  }

  /**
   * Writes the flat {@code .vec}/{@code .vemf} from each field's native host matrix, builds the
   * CAGRA/HNSW graph from the same matrix, then releases the matrix.
   */
  @Override
  public void flush(int maxDoc, DocMap sortMap) throws IOException {
    if (sortMap != null) {
      throw new UnsupportedOperationException(
          "AcceleratedHNSWParams.numInputVectors (native flat buffering) does not support"
              + " index-sorted segments; unset it (0) to enable the sorted flush path");
    }
    for (var field : fields) {
      writeFieldNative(field, maxDoc);
    }
  }

  private void writeFieldNative(NativeFieldWriter fieldData, int maxDoc) throws IOException {
    int count = fieldData.getNativeVectorCount();
    if (count != numInputVectors) {
      throw new IllegalStateException(
          "numInputVectors ("
              + numInputVectors
              + ") must equal the number of vectors added ("
              + count
              + ") for field \""
              + fieldData.fieldInfo().name
              + "\"; the native host matrix is sized for the hint exactly. This usually means"
              + " IndexWriterConfig's auto-flush wasn't disabled (setMaxBufferedDocs /"
              + " setRAMBufferSizeMB(DISABLE_AUTO_FLUSH)), so a flush landed before exactly"
              + " numInputVectors vectors were added; see CagraHnswBulkIndexWriter.");
    }
    FieldInfo fieldInfo = fieldData.fieldInfo();
    try {
      CuVSHostMatrix dataset = fieldData.getHostMatrix();
      nativeFlat.writeField(fieldInfo, dataset, maxDoc, fieldData.getDocsWithFieldSet());
      graphOutput.writeField(fieldInfo, dataset);
    } finally {
      fieldData.releaseNativeBuffer();
    }
  }

  /**
   * Native flat buffering supports only the unsorted single-segment flush path; merges are
   * rejected.
   */
  @Override
  public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    throw new UnsupportedOperationException(
        "AcceleratedHNSWParams.numInputVectors (native flat buffering) supports only the"
            + " unsorted single-segment flush path; unset it (0) to enable merges");
  }

  /**
   * Called once at the end before close.
   */
  @Override
  public void finish() throws IOException {
    if (finished) {
      throw new IllegalStateException("already finished");
    }
    finished = true;
    nativeFlat.finish();
    graphOutput.finish();
  }

  /**
   * Closes the resources.
   */
  @Override
  public void close() throws IOException {
    printInfoStream(infoStream, COMPONENT, "Closing resources");
    IOUtils.close(graphOutput, nativeFlat);
    closeCuVSResourcesInstance();
  }

  /**
   * Returns the memory usage of this object in bytes.
   */
  @Override
  public long ramBytesUsed() {
    long total = SHALLOW_RAM_BYTES_USED;
    for (var field : fields) {
      total += field.ramBytesUsed();
    }
    return total;
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.CuVS2510GPUVectorsFormat.CUVS_INDEX_CODEC_NAME;
import static com.nvidia.cuvs.lucene.CuVS2510GPUVectorsFormat.CUVS_INDEX_EXT;
import static com.nvidia.cuvs.lucene.CuVS2510GPUVectorsFormat.CUVS_META_CODEC_EXT;
import static com.nvidia.cuvs.lucene.CuVS2510GPUVectorsFormat.CUVS_META_CODEC_NAME;
import static com.nvidia.cuvs.lucene.CuVS2510GPUVectorsFormat.VERSION_CURRENT;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.closeCuVSResourcesInstance;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.getCuVSResourcesInstance;
import static com.nvidia.cuvs.lucene.Utils.info;
import static org.apache.lucene.index.VectorEncoding.FLOAT32;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.apache.lucene.util.RamUsageEstimator.shallowSizeOfInstance;

import com.nvidia.cuvs.BruteForceIndex;
import com.nvidia.cuvs.BruteForceIndexParams;
import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CuVSMatrix;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.Objects;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.Sorter.DocMap;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.InfoStream;

/**
 * extends upon KnnVectorsWriter and has implementation for critical methods like flush, merge etc.
 *
 * @since 25.10
 */
public class CuVS2510GPUVectorsWriter extends KnnVectorsWriter {

  private static final long SHALLOW_RAM_BYTES_USED =
      shallowSizeOfInstance(CuVS2510GPUVectorsWriter.class);
  private static final String COMPONENT = "CuVS2510GPUVectorsWriter";
  private static final LuceneProvider LUCENE_PROVIDER;
  private static final List<VectorSimilarityFunction> VECTOR_SIMILARITY_FUNCTIONS;
  private static final int MIN_CAGRA_INDEX_SIZE = 2;

  private final GPUSearchParams gpuSearchParams;
  private final FlatVectorsWriter flatVectorsWriter;
  private final List<GPUFieldWriter> fields = new ArrayList<>();
  private final InfoStream infoStream;
  private IndexOutput meta = null, cuvsIndex = null;
  private boolean finished;

  static {
    try {
      LUCENE_PROVIDER = LuceneProvider.getInstance("99");
      VECTOR_SIMILARITY_FUNCTIONS = LUCENE_PROVIDER.getSimilarityFunctions();
    } catch (Exception e) {
      throw new ExceptionInInitializerError(e.getMessage());
    }
  }

  /**
   * The cuVS index Types.
   */
  public enum IndexType {

    /** Builds a CAGRA index. */
    CAGRA(true, false),

    /** Builds a Brute Force index. */
    BRUTE_FORCE(false, true),

    /** Builds both - CAGRA and Brute Force indexes. */
    CAGRA_AND_BRUTE_FORCE(true, true);
    private final boolean cagra, bruteForce;

    IndexType(boolean cagra, boolean bruteForce) {
      this.cagra = cagra;
      this.bruteForce = bruteForce;
    }

    public boolean isCagra() {
      return cagra;
    }

    public boolean isBruteForce() {
      return bruteForce;
    }
  }

  /**
   * Initializes {@link CuVS2510GPUVectorsWriter}.
   *
   * @param state instance of the SegmentWriteState
   * @param gpuSearchParams An instance of {@link GPUSearchParams}
   * @param flatVectorsWriter instance of FlatVectorsWriter
   *
   * @throws IOException I/O exceptions
   */
  public CuVS2510GPUVectorsWriter(
      SegmentWriteState state, GPUSearchParams gpuSearchParams, FlatVectorsWriter flatVectorsWriter)
      throws IOException {
    super();
    this.gpuSearchParams = gpuSearchParams;
    this.flatVectorsWriter = flatVectorsWriter;
    this.infoStream = state.infoStream;
    String metaFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, CUVS_META_CODEC_EXT);
    String cagraFileName =
        IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, CUVS_INDEX_EXT);
    boolean success = false;
    try {
      meta = state.directory.createOutput(metaFileName, state.context);
      cuvsIndex = state.directory.createOutput(cagraFileName, state.context);
      CodecUtil.writeIndexHeader(
          meta,
          CUVS_META_CODEC_NAME,
          VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      CodecUtil.writeIndexHeader(
          cuvsIndex,
          CUVS_INDEX_CODEC_NAME,
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
   * Add new field for indexing.
   */
  @Override
  public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
    var encoding = fieldInfo.getVectorEncoding();
    if (encoding != FLOAT32) {
      throw new IllegalArgumentException("Expected float32, got:" + encoding);
    }
    var writer = Objects.requireNonNull(flatVectorsWriter.addField(fieldInfo));
    @SuppressWarnings("unchecked")
    var flatWriter = (FlatFieldVectorsWriter<float[]>) writer;
    var cuvsFieldWriter = new GPUFieldWriter(fieldInfo, flatWriter);
    fields.add(cuvsFieldWriter);
    return writer;
  }

  /**
   * Creates CAGRA and/or brute force indexes and writes them.
   *
   * @param fieldInfo Instance of the FieldInFo to use
   * @param vectors list of float vectors to index
   * @throws IOException
   */
  private void writeFieldInternal(FieldInfo fieldInfo, List<float[]> vectors) throws IOException {
    if (vectors != null && vectors.size() == 0) {
      writeEmpty(fieldInfo);
      return;
    }
    long cagraIndexOffset, cagraIndexLength = 0L;
    long bruteForceIndexOffset, bruteForceIndexLength = 0L;

    /*
     * CAGRA has an issue when asked to build an index with just one vector.
     * Hence, we currently fallback to brute force in such a case.
     */
    IndexType indexType =
        gpuSearchParams.getIndexType().isCagra() && vectors.size() < MIN_CAGRA_INDEX_SIZE
            ? IndexType.BRUTE_FORCE
            : gpuSearchParams.getIndexType();

    try {
      cagraIndexOffset = cuvsIndex.getFilePointer();
      if (indexType.isCagra()) {
        var cagraIndexOutputStream = new IndexOutputOutputStream(cuvsIndex);
        try {
          CuVSMatrix cagraDataset =
              Utils.createFloatMatrix(
                  vectors, fieldInfo.getVectorDimension(), getCuVSResourcesInstance());
          writeCagraIndex(cagraIndexOutputStream, cagraDataset);
        } catch (Throwable t) {
          // Fallback to brute force in a few cases, for now.
          // Log it to make it more obvious that this is what is happening.
          info(
              infoStream,
              COMPONENT,
              "CAGRA build failed for field \""
                  + fieldInfo.name
                  + "\", falling back to a brute force index: "
                  + t);
          Utils.handleThrowableWithIgnore(t, t.getMessage());
          indexType = IndexType.BRUTE_FORCE;
        }
        cagraIndexLength = cuvsIndex.getFilePointer() - cagraIndexOffset;
      }
      bruteForceIndexOffset = cuvsIndex.getFilePointer();
      if (indexType.isBruteForce()) {
        var bruteForceIndexOutputStream = new IndexOutputOutputStream(cuvsIndex);
        CuVSMatrix bruteforceDataset =
            Utils.createFloatMatrix(
                vectors, fieldInfo.getVectorDimension(), getCuVSResourcesInstance());

        writeBruteForceIndex(bruteForceIndexOutputStream, bruteforceDataset);
        bruteForceIndexLength = cuvsIndex.getFilePointer() - bruteForceIndexOffset;
      }
      writeMeta(
          fieldInfo,
          vectors.size(),
          cagraIndexOffset,
          cagraIndexLength,
          bruteForceIndexOffset,
          bruteForceIndexLength);
    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
  }

  /**
   * Builds and writes the CAGRA index.
   *
   * @param os Instance of the OutputStream
   * @param dataset The instance of CuVSMatrix holding the dataset
   * @throws Throwable
   */
  private void writeCagraIndex(OutputStream os, CuVSMatrix dataset) throws Throwable {
    CagraIndexParams params =
        CagraIndexParamsFactory.create(gpuSearchParams, dataset.size(), dataset.columns());
    try (CagraIndex index =
            CagraIndex.newBuilder(getCuVSResourcesInstance())
                .withDataset(dataset)
                .withIndexParams(params)
                .build();
        var deviceVectors = dataset.toDevice(getCuVSResourcesInstance())) {
      /*
       * cuVS rejects makePaddedDataset for a device matrix whose rows already sit at the required
       * stride, and asks for a view over that storage instead. Copying would be pointless there
       * anyway, so pick the factory that matches the layout.
       */
      if (CagraIndex.isPaddedDataset(deviceVectors)) {
        try (var indexDatasetView = index.makePaddedDatasetView(deviceVectors)) {
          index.updateDataset(indexDatasetView);
          index.serialize(os);
        }
      } else {
        try (var indexDataset = index.makePaddedDataset(deviceVectors)) {
          index.updateDataset(indexDataset);
          index.serialize(os);
        }
      }
    }
  }

  /**
   * Builds and writes the brute force index.
   *
   * @param os Instance of OutputStream to write the index bytes to
   * @param dataset Instance of CuVSMatrix that holds the data set
   * @throws Throwable
   */
  private void writeBruteForceIndex(OutputStream os, CuVSMatrix dataset) throws Throwable {
    BruteForceIndexParams params =
        new BruteForceIndexParams.Builder()
            .withNumWriterThreads(gpuSearchParams.getWriterThreads())
            .build();
    var index =
        BruteForceIndex.newBuilder(getCuVSResourcesInstance())
            .withIndexParams(params)
            .withDataset(dataset)
            .build();
    index.serialize(os);
    index.close();
  }

  /**
   * Creates the CAGRA and/or brute force indexes and writes them to the disk.
   */
  @Override
  public void flush(int maxDoc, DocMap sortMap) throws IOException {
    flatVectorsWriter.flush(maxDoc, sortMap);
    for (var field : fields) {
      if (sortMap == null) {
        writeField(field);
      } else {
        writeSortingField(field, sortMap);
      }
    }
  }

  /**
   * Calls the method that builds indexes and writes them to the disk.
   *
   * @param fieldData reference to the {@link GPUFieldWriter}
   * @throws IOException
   */
  private void writeField(GPUFieldWriter fieldData) throws IOException {
    writeFieldInternal(fieldData.fieldInfo(), fieldData.getVectors());
  }

  /**
   * Builds indexes and writes them to the disk.
   *
   * @param fieldData reference to the {@link GPUFieldWriter}
   * @param sortMap reference to DocMap
   * @throws IOException I/O Exceptions
   */
  private void writeSortingField(GPUFieldWriter fieldData, Sorter.DocMap sortMap)
      throws IOException {
    DocsWithFieldSet oldDocsWithFieldSet = fieldData.getDocsWithFieldSet();
    final int[] new2OldOrd = new int[oldDocsWithFieldSet.cardinality()];
    mapOldOrdToNewOrd(oldDocsWithFieldSet, sortMap, null, new2OldOrd, null);
    List<float[]> sortedVectors = new ArrayList<float[]>();
    for (int i = 0; i < fieldData.getVectors().size(); i++) {
      sortedVectors.add(fieldData.getVectors().get(new2OldOrd[i]));
    }
    writeFieldInternal(fieldData.fieldInfo(), sortedVectors);
  }

  /**
   * Writes empty meta information for the field.
   *
   * @param fieldInfo instance of the FieldInfo
   * @throws IOException I/O Exceptions
   */
  private void writeEmpty(FieldInfo fieldInfo) throws IOException {
    writeMeta(fieldInfo, 0, 0L, 0L, 0L, 0L);
  }

  /**
   * Writes the meta information for the index.
   *
   * @param field instance of FieldInfo
   * @param count number of vectors
   * @param cagraIndexOffset CAGRA index offset
   * @param cagraIndexLength CAGRA index length
   * @param bruteForceIndexOffset Brute force index offset
   * @param bruteForceIndexLength Brute force index length
   * @throws IOException I/O Exceptions
   */
  private void writeMeta(
      FieldInfo field,
      int count,
      long cagraIndexOffset,
      long cagraIndexLength,
      long bruteForceIndexOffset,
      long bruteForceIndexLength)
      throws IOException {
    meta.writeInt(field.number);
    meta.writeInt(field.getVectorEncoding().ordinal());
    meta.writeInt(distFuncToOrd(field.getVectorSimilarityFunction()));
    meta.writeInt(field.getVectorDimension());
    meta.writeInt(count);
    meta.writeVLong(cagraIndexOffset);
    meta.writeVLong(cagraIndexLength);
    meta.writeVLong(bruteForceIndexOffset);
    meta.writeVLong(bruteForceIndexLength);
  }

  static int distFuncToOrd(VectorSimilarityFunction func) {
    for (int i = 0; i < VECTOR_SIMILARITY_FUNCTIONS.size(); i++) {
      if (VECTOR_SIMILARITY_FUNCTIONS.get(i).equals(func)) {
        return (byte) i;
      }
    }
    throw new IllegalArgumentException("Invalid distance function: " + func);
  }

  /**
   * The inputs the cuVS merge API needs for one field: the CAGRA indexes to concatenate, and the
   * rows of that concatenation that survive the merge.
   *
   * @param readers the readers holding a CAGRA index for the field, in merge order
   * @param rowFilter the rows of the concatenated data sets to keep, or {@code null} to keep all of
   *     them
   * @param mergedVectorCount the number of vectors the merged index will hold
   */
  private record CagraMergeInputs(
      List<CuVS2510GPUVectorsReader> readers, BitSet rowFilter, int mergedVectorCount) {}

  /**
   * Collects the CAGRA indexes that can be handed to the cuVS merge API for this field, together
   * with the filter that keeps the merged rows lined up with the merged flat vectors.
   *
   * @param fieldInfo instance of the FieldInfo
   * @param mergeState instance of the MergeState
   * @return the inputs for the merge, or {@code null} when the cuVS merge API cannot be used for
   *     this field
   * @throws IOException I/O Exceptions
   */
  private CagraMergeInputs cagraMergeInputs(FieldInfo fieldInfo, MergeState mergeState)
      throws IOException {
    // A brute force index cannot be produced by the merge API; it would have to be rebuilt from the
    // vectors on the host anyway, which is what the vector based merge already does.
    if (gpuSearchParams.getIndexType() != IndexType.CAGRA) {
      logNotMergeable(fieldInfo, "the index type is " + gpuSearchParams.getIndexType());
      return null;
    }
    // A sorted merge interleaves the segments' rows instead of concatenating them.
    if (mergeState.needsIndexSort) {
      logNotMergeable(fieldInfo, "the merge has to sort the documents");
      return null;
    }
    List<CuVS2510GPUVectorsReader> readers = new ArrayList<>();
    BitSet rowFilter = new BitSet();
    // Rows of the concatenation the merge API sees, and the ones of those that survive.
    int rowCount = 0;
    int survivingCount = 0;
    for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
      if (KnnVectorsWriter.MergedVectorValues.hasVectorValues(
              mergeState.fieldInfos[i], fieldInfo.name)
          == false) {
        continue;
      }
      KnnVectorsReader knnReader = mergeState.knnVectorsReaders[i];
      if (knnReader instanceof PerFieldKnnVectorsFormat.FieldsReader fieldsReader) {
        knnReader = fieldsReader.getFieldReader(fieldInfo.name);
      }
      if (knnReader == null) {
        continue;
      }
      FloatVectorValues values = knnReader.getFloatVectorValues(fieldInfo.name);
      if (values == null || values.size() == 0) {
        continue;
      }
      if (!(knnReader instanceof CuVS2510GPUVectorsReader cuvsReader)) {
        logNotMergeable(fieldInfo, "a segment is read by a " + knnReader.getClass().getName());
        return null;
      }
      CuVS2510GPUVectorsReader.FieldEntry fieldEntry = cuvsReader.getFieldEntry(fieldInfo.name);
      // A segment too small for CAGRA, or one whose CAGRA build failed, holds a brute force index
      // instead and has nothing to contribute to the merge.
      if (fieldEntry == null || fieldEntry.cagraIndexLength() == 0) {
        logNotMergeable(fieldInfo, "a segment of " + values.size() + " vectors has no CAGRA index");
        return null;
      }
      if (fieldEntry.count() != values.size()) {
        logNotMergeable(
            fieldInfo,
            "a segment holds "
                + fieldEntry.count()
                + " indexed vectors but "
                + values.size()
                + " flat vectors");
        return null;
      }
      BitSet liveRows =
          liveRows(values, fieldEntry.count(), mergeState.liveDocs[i], mergeState.docMaps[i]);
      int liveCount = liveRows.cardinality();
      // A segment whose vectors are all deleted contributes nothing to the merged flat vectors
      // either, so leave it out rather than upload an index only to drop every row of it.
      if (liveCount == 0) {
        continue;
      }
      for (int ord = liveRows.nextSetBit(0); ord >= 0; ord = liveRows.nextSetBit(ord + 1)) {
        rowFilter.set(rowCount + ord);
      }
      rowCount += fieldEntry.count();
      survivingCount += liveCount;
      readers.add(cuvsReader);
    }
    boolean filtered = survivingCount != rowCount;
    // Merging a single index is only worth it when the filter has rows to drop; without one the
    // merge would just copy the index it was given.
    if (readers.isEmpty() || (readers.size() == 1 && filtered == false)) {
      logNotMergeable(fieldInfo, "there is nothing to merge, " + readers.size() + " segments");
      return null;
    }
    // The merged index has to be one CAGRA can build, and cuVS refuses a filter that keeps no rows
    // at all.
    if (survivingCount < MIN_CAGRA_INDEX_SIZE) {
      logNotMergeable(
          fieldInfo, "only " + survivingCount + " vectors survive the deletions of the merge");
      return null;
    }
    // Without deletions every row is kept, and passing no filter lets cuVS skip the gather.
    return new CagraMergeInputs(readers, filtered ? rowFilter : null, survivingCount);
  }

  /**
   * Returns the ordinals of {@code values} whose document survives the merge.
   *
   * <p>A cuVS row id is a vector ordinal rather than a document id, so the deletions have to be
   * translated through {@link KnnVectorValues#ordToDoc}. The test applied to each document is the
   * one {@code DocIDMerger} applies while producing the merged vectors: a document is dropped when
   * the merge maps it to {@code -1}.
   */
  private static BitSet liveRows(
      FloatVectorValues values, int count, Bits liveDocs, MergeState.DocMap docMap) {
    BitSet liveRows = new BitSet(count);
    if (liveDocs == null) {
      liveRows.set(0, count);
      return liveRows;
    }
    for (int ord = 0; ord < count; ord++) {
      if (docMap.get(values.ordToDoc(ord)) != -1) {
        liveRows.set(ord);
      }
    }
    return liveRows;
  }

  /**
   * Reports why this field cannot go through the cuVS merge API and has to fall back to the vector
   * based merge.
   */
  private void logNotMergeable(FieldInfo fieldInfo, String reason) {
    info(
        infoStream,
        COMPONENT,
        "Skipping the cuVS merge API for field \"" + fieldInfo.name + "\": " + reason);
  }

  /**
   * Uses the cuVS API to merge the segments' CAGRA indexes on the device, which avoids copying
   * every vector back to the host and re-uploading it.
   *
   * @param fieldInfo instance of the FieldInfo
   * @param mergeState instance of the MergeState
   * @return true if the field was written, false if the caller has to fall back to the vector
   *     based merge
   * @throws IOException I/O Exceptions
   */
  private boolean mergeCagraIndexes(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    CagraMergeInputs inputs = cagraMergeInputs(fieldInfo, mergeState);
    if (inputs == null) {
      return false;
    }
    long cagraIndexOffset = cuvsIndex.getFilePointer();
    long cagraIndexLength;
    try {
      writeMergedIndexBytes(fieldInfo, inputs);
      cagraIndexLength = cuvsIndex.getFilePointer() - cagraIndexOffset;
    } catch (Throwable t) {
      if (t instanceof Error error) {
        throw error;
      }
      // The merge API needs every input data set on the device at once, so it can run out of
      // device memory where the vector based merge would not. Nothing is committed until the meta
      // entry below is written, and that is what makes the bytes reachable, so falling back here
      // only leaves them unreferenced.
      info(
          infoStream,
          COMPONENT,
          "cuVS merge API failed for field \""
              + fieldInfo.name
              + "\", falling back to a vector based merge: "
              + t);
      return false;
    }
    // The field is committed by this call. Anything that fails from here is a real I/O failure
    // rather than something a rebuild could fix, so it propagates: a fallback now would append a
    // second meta entry for the same field and leave the segment unreadable.
    writeMeta(
        fieldInfo,
        inputs.mergedVectorCount(),
        cagraIndexOffset,
        cagraIndexLength,
        cuvsIndex.getFilePointer(),
        0L);
    info(
        infoStream,
        COMPONENT,
        "Successfully merged "
            + inputs.readers().size()
            + " CAGRA indexes for field \""
            + fieldInfo.name
            + "\" using the cuVS merge API");
    return true;
  }

  /**
   * Merges the field's CAGRA indexes and appends the merged index to the cuVS index output,
   * releasing every index it opened before returning. Writes no meta entry, so a failure leaves
   * nothing but unreferenced bytes behind.
   *
   * @param fieldInfo instance of the FieldInfo
   * @param inputs the indexes to merge and the rows to keep
   * @throws Throwable if the merge, the serialization, or releasing an index fails
   */
  private void writeMergedIndexBytes(FieldInfo fieldInfo, CagraMergeInputs inputs)
      throws Throwable {
    List<CagraIndex> indexes = new ArrayList<>(inputs.readers().size());
    try {
      for (CuVS2510GPUVectorsReader reader : inputs.readers()) {
        indexes.add(reader.openCagraIndexForMerge(fieldInfo.name));
      }
      // Derive the output parameters the same way a build over the merged data set would, so the
      // merged graph degree matches the one a flush of that size produces.
      CagraIndexParams mergeParams =
          CagraIndexParamsFactory.create(
              gpuSearchParams, inputs.mergedVectorCount(), fieldInfo.getVectorDimension());
      try (CagraIndex mergedIndex =
          CagraIndex.merge(
              indexes.toArray(new CagraIndex[indexes.size()]), mergeParams, inputs.rowFilter())) {
        Path tmpFile =
            Files.createTempFile(getCuVSResourcesInstance().tempDirectory(), "mergedindex", "cag");
        try {
          mergedIndex.serialize(new IndexOutputOutputStream(cuvsIndex), tmpFile);
        } finally {
          // cuVS removes the file once it has read it back, but not when serializing failed.
          Files.deleteIfExists(tmpFile);
        }
      }
    } finally {
      // Every index is released even if one of them fails, and the first failure carries the rest
      // so that none of them is lost. IOUtils cannot do this here: CagraIndex.close() throws
      // Exception rather than IOException, so it is not a Closeable.
      Exception failure = null;
      for (CagraIndex index : indexes) {
        try {
          index.close();
        } catch (Exception e) {
          if (failure == null) {
            failure = e;
          } else {
            failure.addSuppressed(e);
          }
        }
      }
      if (failure != null) {
        throw failure;
      }
    }
  }

  /**
   * Creates List<Float[]> from merged vectors.
   */
  private List<float[]> createListFromMergedVectors(FloatVectorValues mergedVectorValues)
      throws IOException {
    List<float[]> res = new ArrayList<float[]>();
    KnnVectorValues.DocIndexIterator iter = mergedVectorValues.iterator();
    for (int docV = iter.nextDoc(); docV != NO_MORE_DOCS; docV = iter.nextDoc()) {
      int ordinal = iter.index();
      float[] vector = mergedVectorValues.vectorValue(ordinal);
      res.add(vector.clone());
    }
    return res;
  }

  /**
   * Fallback method that rebuilds indexes from merged vectors.
   * Used when native CAGRA merge() is not possible. Also used
   * when non-CAGRA index types are used (for e.g. Brute Force index).
   */
  private void vectorBasedMerge(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    try {
      List<float[]> dataset =
          createListFromMergedVectors(
              KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState));
      writeFieldInternal(fieldInfo, dataset);
    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
  }

  /**
   * Write field for merging.
   */
  @Override
  public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    flatVectorsWriter.mergeOneField(fieldInfo, mergeState);
    if (mergeCagraIndexes(fieldInfo, mergeState) == false) {
      vectorBasedMerge(fieldInfo, mergeState);
    }
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

  /**
   * Called once at the end before close.
   */
  @Override
  public void finish() throws IOException {
    if (finished) {
      throw new IllegalStateException("already finished");
    }
    finished = true;
    flatVectorsWriter.finish();
    if (meta != null) {
      // write end of fields marker
      meta.writeInt(-1);
      CodecUtil.writeFooter(meta);
    }
    if (cuvsIndex != null) {
      CodecUtil.writeFooter(cuvsIndex);
    }
  }

  /**
   * Close the applicable resources.
   */
  @Override
  public void close() throws IOException {
    IOUtils.close(meta, cuvsIndex, flatVectorsWriter);
    closeCuVSResourcesInstance();
  }
}

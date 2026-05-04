/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
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
import java.util.List;
import java.util.Objects;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.Sorter.DocMap;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.internal.hppc.IntObjectHashMap;
import org.apache.lucene.store.IndexOutput;
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
    CagraIndex index =
        CagraIndex.newBuilder(getCuVSResourcesInstance())
            .withDataset(dataset)
            .withIndexParams(params)
            .build();
    index.serialize(os);
    index.close();
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
   * Uses the cuVS API to merge CAGRA indexes.
   *
   * This is currently (and intentionally) marked as unused and will be plugged in later.
   *
   * @param fieldInfo instance of the FieldInfo
   * @param mergeState instance of the MergeState
   * @throws IOException I/O Exceptions
   */
  @SuppressWarnings("unused")
  private void mergeCagraIndexes(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    try {
      List<CagraIndex> cagraIndexes = new ArrayList<>();
      // We need this count so that the merged segment's meta information has the vector count.
      int totalVectorCount = 0;
      for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
        KnnVectorsReader knnReader = mergeState.knnVectorsReaders[i];
        // Access the CAGRA index for this field from the reader
        if (knnReader != null) {
          if (knnReader instanceof CuVS2510GPUVectorsReader cvr) {
            if (cvr != null) {
              totalVectorCount += cvr.getFieldEntries().get(fieldInfo.number).count();
              CagraIndex cagraIndex = getCagraIndexFromReader(cvr, fieldInfo.name);
              if (cagraIndex != null) {
                cagraIndexes.add(cagraIndex);
              }
            }
          } else {
            // This should never happen
            throw new RuntimeException(
                "Reader is not of CuVSVectorsReader type. Instead it is: " + knnReader.getClass());
          }
        }
      }
      assert cagraIndexes.size() > 1;
      CagraIndex mergedIndex =
          CagraIndex.merge(cagraIndexes.toArray(new CagraIndex[cagraIndexes.size()]));
      writeMergedCagraIndex(fieldInfo, mergedIndex, totalVectorCount);
      info(
          infoStream,
          COMPONENT,
          "Successfully merged " + cagraIndexes.size() + " CAGRA indexes using native merge API");
    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
  }

  /**
   * Extracts the CAGRA index for a specific field from a CuVSVectorsReader.
   */
  private CagraIndex getCagraIndexFromReader(CuVS2510GPUVectorsReader reader, String fieldName) {
    try {
      IntObjectHashMap<GPUIndex> cuvsIndices = reader.getCuvsIndexes();
      FieldInfos fieldInfos = reader.getFieldInfos();
      FieldInfo fieldInfo = fieldInfos.fieldInfo(fieldName);
      if (fieldInfo != null) {
        GPUIndex cuvsIndex = cuvsIndices.get(fieldInfo.number);
        if (cuvsIndex != null) {
          return cuvsIndex.getCagraIndex();
        }
      }
    } catch (Exception e) {
      info(
          infoStream,
          COMPONENT,
          "Failed to extract CAGRA index for field " + fieldName + ": " + e.getMessage());
      throw e;
    }
    return null;
  }

  /**
   * Writes a pre-built merged CAGRA index to the output.
   */
  private void writeMergedCagraIndex(FieldInfo fieldInfo, CagraIndex mergedIndex, int vectorCount)
      throws IOException {
    try {
      long cagraIndexOffset = cuvsIndex.getFilePointer();
      var cagraIndexOutputStream = new IndexOutputOutputStream(cuvsIndex);
      Path tmpFile =
          Files.createTempFile(getCuVSResourcesInstance().tempDirectory(), "mergedindex", "cag");
      mergedIndex.serialize(cagraIndexOutputStream, tmpFile);
      long cagraIndexLength = cuvsIndex.getFilePointer() - cagraIndexOffset;
      writeMeta(fieldInfo, vectorCount, cagraIndexOffset, cagraIndexLength, 0L, 0L);
      mergedIndex.close();
    } catch (Throwable t) {
      Utils.handleThrowable(t);
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
    vectorBasedMerge(fieldInfo, mergeState);
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

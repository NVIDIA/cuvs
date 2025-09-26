/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.CuVS2510GPUVectorsFormat.CUVS_INDEX_CODEC_NAME;
import static com.nvidia.cuvs.lucene.CuVS2510GPUVectorsFormat.CUVS_INDEX_EXT;
import static com.nvidia.cuvs.lucene.CuVS2510GPUVectorsFormat.CUVS_META_CODEC_EXT;
import static com.nvidia.cuvs.lucene.CuVS2510GPUVectorsFormat.CUVS_META_CODEC_NAME;
import static com.nvidia.cuvs.lucene.CuVS2510GPUVectorsFormat.VERSION_CURRENT;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.SIMILARITY_FUNCTIONS;
import static org.apache.lucene.index.VectorEncoding.FLOAT32;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.apache.lucene.util.RamUsageEstimator.shallowSizeOfInstance;

import com.nvidia.cuvs.BruteForceIndex;
import com.nvidia.cuvs.BruteForceIndexParams;
import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSResources;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.logging.Logger;
import java.util.stream.IntStream;
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
 * KnnVectorsWriter for CuVS, responsible for merge and flush of vectors into
 * GPU
 */
public class CuVS2510GPUVectorsWriter extends KnnVectorsWriter {

  private static final long SHALLOW_RAM_BYTES_USED =
      shallowSizeOfInstance(CuVS2510GPUVectorsWriter.class);

  @SuppressWarnings("unused")
  private static final Logger log = Logger.getLogger(CuVS2510GPUVectorsWriter.class.getName());

  /** The name of the CUVS component for the info-stream * */
  private static final String CUVS_COMPONENT = "CUVS";

  // The minimum number of vectors in the dataset required before
  // we attempt to build a Cagra index
  static final int MIN_CAGRA_INDEX_SIZE = 2;

  private final int cuvsWriterThreads;
  private final int intGraphDegree;
  private final int graphDegree;

  private final CuVSResources resources;
  private final IndexType indexType;

  private final FlatVectorsWriter flatVectorsWriter; // for writing the raw vectors
  private final List<GPUFieldWriter> fields = new ArrayList<>();
  private IndexOutput meta = null, cuvsIndex = null;
  private IndexOutput hnswMeta = null, hnswVectorIndex = null;
  private final InfoStream infoStream;
  private boolean finished;

  /** The CuVS index Type. */
  public enum IndexType {

    /** Builds a Cagra index. */
    CAGRA(true, false, false),

    /** Builds a Brute Force index. */
    BRUTE_FORCE(false, true, false),

    /** Builds an HSNW index - suitable for searching on CPU. */
    HNSW(false, false, true),

    /** Builds a Cagra and a Brute Force index. */
    CAGRA_AND_BRUTE_FORCE(true, true, false);
    private final boolean cagra, bruteForce, hnsw;

    IndexType(boolean cagra, boolean bruteForce, boolean hnsw) {
      this.cagra = cagra;
      this.bruteForce = bruteForce;
      this.hnsw = hnsw;
    }

    public boolean cagra() {
      return cagra;
    }

    public boolean bruteForce() {
      return bruteForce;
    }

    public boolean hnsw() {
      return hnsw;
    }
  }

  public CuVS2510GPUVectorsWriter(
      SegmentWriteState state,
      int cuvsWriterThreads,
      int intGraphDegree,
      int graphDegree,
      int hnswLayers,
      IndexType indexType,
      CuVSResources resources,
      FlatVectorsWriter flatVectorsWriter)
      throws IOException {
    super();
    this.indexType = indexType;
    this.cuvsWriterThreads = cuvsWriterThreads;
    this.intGraphDegree = intGraphDegree;
    this.graphDegree = graphDegree;
    this.resources = resources;
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

  @Override
  public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
    var encoding = fieldInfo.getVectorEncoding();
    if (encoding != FLOAT32) {
      throw new IllegalArgumentException("expected float32, got:" + encoding);
    }
    var writer = Objects.requireNonNull(flatVectorsWriter.addField(fieldInfo));
    @SuppressWarnings("unchecked")
    var flatWriter = (FlatFieldVectorsWriter<float[]>) writer;
    var cuvsFieldWriter = new GPUFieldWriter(fieldInfo, flatWriter);
    fields.add(cuvsFieldWriter);
    return writer;
  }

  static String indexMsg(int size, int... args) {
    StringBuilder sb = new StringBuilder("cagra index params");
    sb.append(": size=").append(size);
    sb.append(", intGraphDegree=").append(args[0]);
    sb.append(", actualIntGraphDegree=").append(args[1]);
    sb.append(", graphDegree=").append(args[2]);
    sb.append(", actualGraphDegree=").append(args[3]);
    return sb.toString();
  }

  private CagraIndexParams cagraIndexParams(int size) {
    if (size < 2) {
      // https://github.com/rapidsai/cuvs/issues/666
      throw new IllegalArgumentException("cagra index must be greater than 2");
    }

    return new CagraIndexParams.Builder()
        .withNumWriterThreads(cuvsWriterThreads)
        .withIntermediateGraphDegree(intGraphDegree)
        .withGraphDegree(graphDegree)
        .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
        .build();
  }

  private void info(String msg) {
    if (infoStream.isEnabled(CUVS_COMPONENT)) {
      infoStream.message(CUVS_COMPONENT, msg);
    }
  }

  private void writeFieldInternal(FieldInfo fieldInfo, List<float[]> vectors) throws IOException {
    if (vectors.size() == 0) {
      writeEmpty(fieldInfo);
      return;
    }
    long cagraIndexOffset, cagraIndexLength = 0L;
    long bruteForceIndexOffset, bruteForceIndexLength = 0L;
    long hnswIndexOffset, hnswIndexLength = 0L;

    // workaround for the minimum number of vectors for Cagra
    IndexType indexType =
        this.indexType.cagra() && vectors.size() < MIN_CAGRA_INDEX_SIZE
            ? IndexType.BRUTE_FORCE
            : this.indexType;

    try {

      cagraIndexOffset = cuvsIndex.getFilePointer();
      if (indexType.cagra()) {
        try {
          var cagraIndexOutputStream = new IndexOutputOutputStream(cuvsIndex);
          CuVSMatrix dataset = Utils.createFloatMatrix(vectors, fieldInfo.getVectorDimension());
          writeCagraIndex(cagraIndexOutputStream, dataset);
        } catch (Throwable t) {
          Utils.handleThrowableWithIgnore(t, CANNOT_GENERATE_CAGRA);
          // workaround for cuVS issue
          indexType = IndexType.BRUTE_FORCE;
        }
        cagraIndexLength = cuvsIndex.getFilePointer() - cagraIndexOffset;
      }

      bruteForceIndexOffset = cuvsIndex.getFilePointer();
      if (indexType.bruteForce()) {
        var bruteForceIndexOutputStream = new IndexOutputOutputStream(cuvsIndex);
        CuVSMatrix dataset = Utils.createFloatMatrix(vectors, fieldInfo.getVectorDimension());
        writeBruteForceIndex(bruteForceIndexOutputStream, dataset);
        bruteForceIndexLength = cuvsIndex.getFilePointer() - bruteForceIndexOffset;
      }

      hnswIndexOffset = cuvsIndex.getFilePointer();
      if (indexType.hnsw()) {
        var hnswIndexOutputStream = new IndexOutputOutputStream(cuvsIndex);
        if (vectors.size() > MIN_CAGRA_INDEX_SIZE) {
          try {
            CuVSMatrix dataset = Utils.createFloatMatrix(vectors, fieldInfo.getVectorDimension());
            writeHNSWIndex(hnswIndexOutputStream, dataset);
          } catch (Throwable t) {
            Utils.handleThrowableWithIgnore(t, CANNOT_GENERATE_CAGRA);
          }
        }
        hnswIndexLength = cuvsIndex.getFilePointer() - hnswIndexOffset;
      }

      // Only write meta for non-HNSW_LUCENE modes
      writeMeta(
          fieldInfo,
          vectors.size(),
          cagraIndexOffset,
          cagraIndexLength,
          bruteForceIndexOffset,
          bruteForceIndexLength,
          hnswIndexOffset,
          hnswIndexLength);
    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
  }

  private void writeCagraIndex(OutputStream os, CuVSMatrix dataset) throws Throwable {
    if (dataset.size() < 2) {
      throw new IllegalArgumentException(dataset.size() + " vectors, less than min [2] required");
    }
    CagraIndexParams params = cagraIndexParams((int) dataset.size());
    long startTime = System.nanoTime();
    CagraIndex index =
        CagraIndex.newBuilder(resources).withDataset(dataset).withIndexParams(params).build();
    long elapsedMillis = Utils.nanosToMillis(System.nanoTime() - startTime);
    info("Cagra index created in " + elapsedMillis + "ms, with " + dataset.size() + " vectors");
    Path tmpFile = Files.createTempFile(resources.tempDirectory(), "tmpindex", "cag");
    index.serialize(os, tmpFile);
    index.close();
  }

  private void writeBruteForceIndex(OutputStream os, CuVSMatrix dataset) throws Throwable {
    BruteForceIndexParams params =
        new BruteForceIndexParams.Builder()
            .withNumWriterThreads(32) // TODO: Make this configurable.
            .build();
    long startTime = System.nanoTime();
    var index =
        BruteForceIndex.newBuilder(resources).withIndexParams(params).withDataset(dataset).build();
    long elapsedMillis = Utils.nanosToMillis(System.nanoTime() - startTime);
    info("bf index created in " + elapsedMillis + "ms, with " + dataset.size() + " vectors");
    index.serialize(os);
    index.close();
  }

  private void writeHNSWIndex(OutputStream os, CuVSMatrix dataset) throws Throwable {
    if (dataset.size() < 2) {
      throw new IllegalArgumentException(dataset.size() + " vectors, less than min [2] required");
    }
    CagraIndexParams indexParams = cagraIndexParams((int) dataset.size());
    long startTime = System.nanoTime();
    CagraIndex index =
        CagraIndex.newBuilder(resources).withDataset(dataset).withIndexParams(indexParams).build();
    long elapsedMillis = Utils.nanosToMillis(System.nanoTime() - startTime);
    info("HNSW index created in " + elapsedMillis + "ms, with " + dataset.size() + " vectors");
    Path tmpFile = Files.createTempFile("tmpindex", "hnsw");
    index.serializeToHNSW(os, tmpFile);
    index.close();
  }

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

  private void writeField(GPUFieldWriter fieldData) throws IOException {
    writeFieldInternal(fieldData.fieldInfo(), fieldData.getVectors());
  }

  private void writeSortingField(GPUFieldWriter fieldData, Sorter.DocMap sortMap)
      throws IOException {

    DocsWithFieldSet oldDocsWithFieldSet = fieldData.getDocsWithFieldSet();
    final int[] new2OldOrd = new int[oldDocsWithFieldSet.cardinality()]; // new ord to old ord
    mapOldOrdToNewOrd(oldDocsWithFieldSet, sortMap, null, new2OldOrd, null);

    List<float[]> sortedVectors = new ArrayList<float[]>();
    for (int i = 0; i < fieldData.getVectors().size(); i++) {
      sortedVectors.add(fieldData.getVectors().get(new2OldOrd[i]));
    }

    writeFieldInternal(fieldData.fieldInfo(), sortedVectors);
  }

  private void writeEmpty(FieldInfo fieldInfo) throws IOException {
    writeMeta(fieldInfo, 0, 0L, 0L, 0L, 0L, 0L, 0L);
  }

  private void writeMeta(
      FieldInfo field,
      int count,
      long cagraIndexOffset,
      long cagraIndexLength,
      long bruteForceIndexOffset,
      long bruteForceIndexLength,
      long hnswIndexOffset,
      long hnswIndexLength)
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
    meta.writeVLong(hnswIndexOffset);
    meta.writeVLong(hnswIndexLength);
  }

  static int distFuncToOrd(VectorSimilarityFunction func) {
    for (int i = 0; i < SIMILARITY_FUNCTIONS.size(); i++) {
      if (SIMILARITY_FUNCTIONS.get(i).equals(func)) {
        return (byte) i;
      }
    }
    throw new IllegalArgumentException("invalid distance function: " + func);
  }

  // We currently ignore this, until cuVS supports tiered indices
  private static final String CANNOT_GENERATE_CAGRA =
      """
      Could not generate an intermediate CAGRA graph because the initial \
      kNN graph contains too many invalid or duplicated neighbor nodes. \
      This error can occur, for example, if too many overflows occur \
      during the norm computation between the dataset vectors\
      """;

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
      info("Successfully merged " + cagraIndexes.size() + " CAGRA indexes using native merge API");

    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
  }

  /**
   * Creates List<Float[]> from merged vectors
   * */
  private List<float[]> createListFromMergedVectors(FloatVectorValues mergedVectorValues)
      throws IOException {
    List<float[]> res = new ArrayList<float[]>();
    KnnVectorValues.DocIndexIterator iter = mergedVectorValues.iterator();
    for (int docV = iter.nextDoc(); docV != NO_MORE_DOCS; docV = iter.nextDoc()) {
      int ordinal = iter.index();
      float[] vector = mergedVectorValues.vectorValue(ordinal);
      res.add(vector);
    }
    return res;
  }

  /**
   * Fallback method that rebuilds indexes from merged vectors.
   * Used when native CAGRA merge() is not possible. Also used
   * when non-CAGRA index types are used (for e.g. Brute Force index).
   */
  private void vectorBasedMerge(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    if (fieldInfo.getVectorEncoding() != FLOAT32) {
      throw new AssertionError("Only Float32 supported");
    }
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
      e.printStackTrace();
      info("Failed to extract CAGRA index for field " + fieldName + ": " + e.getMessage());
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

      // Serialize the merged index
      Path tmpFile = Files.createTempFile(resources.tempDirectory(), "mergedindex", "cag");
      mergedIndex.serialize(cagraIndexOutputStream, tmpFile);
      long cagraIndexLength = cuvsIndex.getFilePointer() - cagraIndexOffset;

      writeMeta(fieldInfo, vectorCount, cagraIndexOffset, cagraIndexLength, 0L, 0L, 0L, 0L);

      // Clean up the merged index
      mergedIndex.close();
    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
  }

  @Override
  public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    flatVectorsWriter.mergeOneField(fieldInfo, mergeState);

    if (indexType.cagra() && !indexType.bruteForce()) {
      // Since CAGRA merge does not support merging of indexes with purging of deletes,
      // we fallback to vector-based re-indexing. Issue:
      // https://github.com/rapidsai/cuvs/issues/1253
      boolean hasDeletions =
          IntStream.range(0, mergeState.liveDocs.length)
              .anyMatch(
                  i ->
                      mergeState.liveDocs[i] == null
                          || IntStream.range(0, mergeState.maxDocs[i])
                              .anyMatch(j -> !mergeState.liveDocs[i].get(j)));

      if (mergeState.knnVectorsReaders.length > 1 && !hasDeletions) {
        mergeCagraIndexes(fieldInfo, mergeState);
      } else {
        // CAGRA's merge API does not handle the trivial case of merging 1 index.
        vectorBasedMerge(fieldInfo, mergeState);
      }

    } else {
      // If there is a Brute Force index then re-index using the vectors even if there is a CAGRA
      // index.
      vectorBasedMerge(fieldInfo, mergeState);
    }
  }

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

    {
      if (hnswMeta != null) {
        // write end of fields marker
        hnswMeta.writeInt(-1);
        CodecUtil.writeFooter(hnswMeta);
      }
      if (hnswVectorIndex != null) {
        CodecUtil.writeFooter(hnswVectorIndex);
      }
    }
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(meta, cuvsIndex, hnswMeta, hnswVectorIndex, flatVectorsWriter);
  }

  @Override
  public long ramBytesUsed() {
    long total = SHALLOW_RAM_BYTES_USED;
    for (var field : fields) {
      total += field.ramBytesUsed();
    }
    return total;
  }
}

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
import static com.nvidia.cuvs.lucene.CuVS2510GPUVectorsFormat.VERSION_START;

import com.nvidia.cuvs.BruteForceIndex;
import com.nvidia.cuvs.BruteForceQuery;
import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraQuery;
import com.nvidia.cuvs.CagraSearchParams;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSResources;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.internal.hppc.IntObjectHashMap;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.DataInput;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.ReadAdvice;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.hnsw.IntToIntFunction;

/**
 * KnnVectorsReader instance associated with cuVS format for reading vectors from an index.
 *
 * @since 25.10
 */
public class CuVS2510GPUVectorsReader extends KnnVectorsReader {

  @SuppressWarnings("unused")
  private static final Logger log = Logger.getLogger(CuVS2510GPUVectorsReader.class.getName());

  private final CuVSResources resources;
  private final FlatVectorsReader flatVectorsReader; // for reading the raw vectors
  private final FieldInfos fieldInfos;
  private final IntObjectHashMap<FieldEntry> fields;
  private final IntObjectHashMap<GPUIndex> cuvsIndices;
  private final IndexInput cuvsIndexInput;

  /**
   * Initializes the {@link CuVS2510GPUVectorsReader}, checks and loads the index.
   *
   * @param state instance of the SegmentReadState
   * @param resources instance of the CuVSResources
   * @param flatReader instance of the FlatVectorsReader
   *
   * @throws IOException I/O exception
   */
  public CuVS2510GPUVectorsReader(
      SegmentReadState state, CuVSResources resources, FlatVectorsReader flatReader)
      throws IOException {
    this.resources = resources;
    this.flatVectorsReader = flatReader;
    this.fieldInfos = state.fieldInfos;
    this.fields = new IntObjectHashMap<>();

    String metaFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, CUVS_META_CODEC_EXT);
    boolean success = false;
    int versionMeta = -1;
    try (ChecksumIndexInput meta = state.directory.openChecksumInput(metaFileName)) {
      Throwable priorException = null;
      try {
        versionMeta =
            CodecUtil.checkIndexHeader(
                meta,
                CUVS_META_CODEC_NAME,
                VERSION_START,
                VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix);
        readFields(meta);
      } catch (Throwable exception) {
        priorException = exception;
      } finally {
        CodecUtil.checkFooter(meta, priorException);
      }
      var ioContext = state.context.withReadAdvice(ReadAdvice.SEQUENTIAL);
      cuvsIndexInput = openCuVSInput(state, versionMeta, ioContext);
      cuvsIndices = loadCuVSIndices();
      success = true;
    } finally {
      if (success == false) {
        IOUtils.closeWhileHandlingException(this);
      }
    }
  }

  /**
   * Opens and returns the IndexInput for the segment file.
   *
   * @param state instance of the SegmentReadState
   * @param versionMeta the version number
   * @param context instance of the IOContext
   * @return an instance of the IndexInput
   * @throws IOException
   */
  private static IndexInput openCuVSInput(
      SegmentReadState state, int versionMeta, IOContext context) throws IOException {
    String fileName =
        IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, CUVS_INDEX_EXT);
    IndexInput in = state.directory.openInput(fileName, context);
    boolean success = false;
    try {
      int versionVectorData =
          CodecUtil.checkIndexHeader(
              in,
              CUVS_INDEX_CODEC_NAME,
              VERSION_START,
              VERSION_CURRENT,
              state.segmentInfo.getId(),
              state.segmentSuffix);
      checkVersion(versionMeta, versionVectorData, in);
      CodecUtil.retrieveChecksum(in);
      success = true;
      return in;
    } finally {
      if (success == false) {
        IOUtils.closeWhileHandlingException(in);
      }
    }
  }

  /**
   * Confirms that the vector dimensions are as expected.
   *
   * @param info instance of the FieldInfo that describes document fields
   * @param fieldEntry instance of the FieldEntry that holds the meta information for the field
   */
  private void validateFieldEntry(FieldInfo info, FieldEntry fieldEntry) {
    int dimension = info.getVectorDimension();
    if (dimension != fieldEntry.dims()) {
      throw new IllegalStateException(
          "Inconsistent vector dimension for field=\""
              + info.name
              + "\"; "
              + dimension
              + " != "
              + fieldEntry.dims());
    }
  }

  /**
   * Reads the fieldInfo for each index field and loads FieldEntry in a map.
   *
   * @param meta intance of the ChecksumIndexInput
   * @throws IOException
   */
  private void readFields(ChecksumIndexInput meta) throws IOException {
    for (int fieldNumber = meta.readInt(); fieldNumber != -1; fieldNumber = meta.readInt()) {
      FieldInfo info = fieldInfos.fieldInfo(fieldNumber);
      if (info == null) {
        throw new CorruptIndexException("Invalid field number: " + fieldNumber, meta);
      }
      FieldEntry fieldEntry = readField(meta, info);
      validateFieldEntry(info, fieldEntry);
      fields.put(info.number, fieldEntry);
    }
  }

  // List of vector similarity functions. This list is defined here, in order
  // to avoid an undesirable dependency on the declaration and order of values
  // in VectorSimilarityFunction. The list values and order must be identical
  // to that of {@link o.a.l.c.l.Lucene94FieldInfosFormat#SIMILARITY_FUNCTIONS}.
  static final List<VectorSimilarityFunction> SIMILARITY_FUNCTIONS =
      List.of(
          VectorSimilarityFunction.EUCLIDEAN,
          VectorSimilarityFunction.DOT_PRODUCT,
          VectorSimilarityFunction.COSINE,
          VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT);

  /**
   * Checks the distance function validity and returns it.
   *
   * @param input instance of DataInput
   * @return an instance of VectorSimilarityFunction
   * @throws IOException
   */
  static VectorSimilarityFunction readSimilarityFunction(DataInput input) throws IOException {
    int i = input.readInt();
    if (i < 0 || i >= SIMILARITY_FUNCTIONS.size()) {
      throw new IllegalArgumentException("invalid distance function: " + i);
    }
    return SIMILARITY_FUNCTIONS.get(i);
  }

  /**
   * Reads the vector encoding (The numeric datatype of the vector values) from the DataInput.
   *
   * @param input instance of DataInput
   * @return the vector encoding
   * @throws IOException
   */
  static VectorEncoding readVectorEncoding(DataInput input) throws IOException {
    int encodingId = input.readInt();
    if (encodingId < 0 || encodingId >= VectorEncoding.values().length) {
      throw new CorruptIndexException("Invalid vector encoding id: " + encodingId, input);
    }
    return VectorEncoding.values()[encodingId];
  }

  /**
   * Reads the field from IndexInput using FieldInfo.
   *
   * @param input instance of IndexInput
   * @param info instance of FieldInfo
   * @return the field entry
   * @throws IOException
   */
  private FieldEntry readField(IndexInput input, FieldInfo info) throws IOException {
    VectorEncoding vectorEncoding = readVectorEncoding(input);
    VectorSimilarityFunction similarityFunction = readSimilarityFunction(input);
    if (similarityFunction != info.getVectorSimilarityFunction()) {
      throw new IllegalStateException(
          "Inconsistent vector similarity function for field=\""
              + info.name
              + "\"; "
              + similarityFunction
              + " != "
              + info.getVectorSimilarityFunction());
    }
    return FieldEntry.readEntry(input, vectorEncoding, info.getVectorSimilarityFunction());
  }

  /**
   * Gets the FieldEntry from the map using the field name. Check the encoding as well.
   *
   * @param field name of the field
   * @param expectedEncoding expected encoding
   * @return an instance of FieldEntry that Holds the meta information for the field
   */
  private FieldEntry getFieldEntry(String field, VectorEncoding expectedEncoding) {
    final FieldInfo info = fieldInfos.fieldInfo(field);
    final FieldEntry fieldEntry;
    if (info == null || (fieldEntry = fields.get(info.number)) == null) {
      throw new IllegalArgumentException("field=\"" + field + "\" not found");
    }
    if (fieldEntry.vectorEncoding != expectedEncoding) {
      throw new IllegalArgumentException(
          "field=\""
              + field
              + "\" is encoded as: "
              + fieldEntry.vectorEncoding
              + " expected: "
              + expectedEncoding);
    }
    return fieldEntry;
  }

  /**
   * Invokes loadCuVSIndex for each field and returns the map of {@link GPUIndex}.
   *
   * @return the map containing {@link GPUIndex} objects
   * @throws IOException
   */
  private IntObjectHashMap<GPUIndex> loadCuVSIndices() throws IOException {
    var indices = new IntObjectHashMap<GPUIndex>();
    for (var e : fields) {
      var fieldEntry = e.value;
      int fieldNumber = e.key;
      var cuvsIndex = loadCuVSIndex(fieldEntry);
      indices.put(fieldNumber, cuvsIndex);
    }
    return indices;
  }

  /**
   * Loads the CAGRA and bruteforce index (if exists) onto the GPU.
   *
   * @param fieldEntry instance of {@link FieldEntry}
   * @return return the instance of {@link GPUIndex}
   * @throws IOException
   */
  private GPUIndex loadCuVSIndex(FieldEntry fieldEntry) throws IOException {
    CagraIndex cagraIndex = null;
    BruteForceIndex bruteForceIndex = null;

    try {
      long len = fieldEntry.cagraIndexLength();
      if (len > 0) {
        long off = fieldEntry.cagraIndexOffset();
        try (var slice = cuvsIndexInput.slice("cagra index", off, len);
            var in = new IndexInputInputStream(slice)) {
          cagraIndex = CagraIndex.newBuilder(resources).from(in).build();
        }
      }

      len = fieldEntry.bruteForceIndexLength();
      if (len > 0) {
        long off = fieldEntry.bruteForceIndexOffset();
        try (var slice = cuvsIndexInput.slice("bf index", off, len);
            var in = new IndexInputInputStream(slice)) {
          bruteForceIndex = BruteForceIndex.newBuilder(resources).from(in).build();
        }
      }
    } catch (Throwable t) {
      Utils.handleThrowable(t);
    }
    return new GPUIndex(cagraIndex, bruteForceIndex);
  }

  /**
   * Closes the resources.
   */
  @Override
  public void close() throws IOException {
    var closeableStream =
        Stream.concat(
            Stream.of(flatVectorsReader, cuvsIndexInput),
            stream(cuvsIndices.values().iterator()).map(cursor -> cursor.value));
    IOUtils.close(closeableStream::iterator);
  }

  static <T> Stream<T> stream(Iterator<T> iterator) {
    return StreamSupport.stream(((Iterable<T>) () -> iterator).spliterator(), false);
  }

  /**
   * Checks consistency of this reader.
   */
  @Override
  public void checkIntegrity() throws IOException {
    // TODO: Pending implementation
  }

  /**
   * Returns the FloatVectorValues for the given field.
   */
  @Override
  public FloatVectorValues getFloatVectorValues(String field) throws IOException {
    return flatVectorsReader.getFloatVectorValues(field);
  }

  /**
   * Returns the FloatVectorValues for the given field.
   *
   * This is not supported.
   */
  @Override
  public ByteVectorValues getByteVectorValues(String field) {
    throw new UnsupportedOperationException("byte vectors not supported");
  }

  /** Native float to float function */
  public interface FloatToFloatFunction {
    float apply(float v);
  }

  /**
   * Returns a long array from bits.
   */
  static long[] bitsToLongArray(Bits bits) {
    if (bits instanceof FixedBitSet fixedBitSet) {
      return fixedBitSet.getBits();
    } else {
      return FixedBitSet.copyOf(bits).getBits();
    }
  }

  /**
   * Get the score normalization function.
   *
   * @param sim instance of VectorSimilarityFunction
   * @return an instance of the FloatToFloatFunction
   */
  static FloatToFloatFunction getScoreNormalizationFunc(VectorSimilarityFunction sim) {
    // TODO: check for different similarities
    return score -> (1f / (1f + score));
  }

  // This is a hack - https://github.com/rapidsai/cuvs/issues/696
  static final int FILTER_OVER_SAMPLE = 10;

  /**
   * Returns the k nearest neighbor documents using cuVS's CAGRA or Bruteforce algorithm for this field, to the given vector.
   */
  @Override
  public void search(String field, float[] target, KnnCollector knnCollector, Bits acceptDocs)
      throws IOException {
    var fieldEntry = getFieldEntry(field, VectorEncoding.FLOAT32);
    if (fieldEntry.count() == 0 || knnCollector.k() == 0) {
      return;
    }

    var fieldNumber = fieldInfos.fieldInfo(field).number;

    GPUIndex cuvsIndex = cuvsIndices.get(fieldNumber);
    if (cuvsIndex == null) {
      throw new IllegalStateException("not index found for field:" + field);
    }

    int collectorTopK = knnCollector.k();
    if (acceptDocs != null) {
      collectorTopK = knnCollector.k() * FILTER_OVER_SAMPLE;
    }
    final int topK = Math.min(collectorTopK, fieldEntry.count());
    assert topK > 0 : "Expected topK > 0, got:" + topK;

    Map<Integer, Float> result;
    if (knnCollector.k() <= 1024 && cuvsIndex.getCagraIndex() != null) {
      CagraSearchParams searchParams =
          new CagraSearchParams.Builder()
              .withItopkSize(topK) // TODO: params
              .withSearchWidth(1)
              .build();

      var query =
          new CagraQuery.Builder(resources)
              .withTopK(topK)
              .withSearchParams(searchParams)
              .withQueryVectors(CuVSMatrix.ofArray(new float[][] {target}))
              .build();

      CagraIndex cagraIndex = cuvsIndex.getCagraIndex();
      List<Map<Integer, Float>> searchResult = null;
      try {
        searchResult = cagraIndex.search(query).getResults();
      } catch (Throwable t) {
        Utils.handleThrowable(t);
      }
      // List expected to have only one entry because of single query "target".
      assert searchResult.size() == 1;
      result = searchResult.getFirst();
    } else {
      BruteForceIndex bruteforceIndex = cuvsIndex.getBruteforceIndex();
      assert bruteforceIndex != null;
      var queryBuilder =
          new BruteForceQuery.Builder(resources)
              .withQueryVectors(new float[][] {target})
              .withTopK(topK);
      BruteForceQuery query = queryBuilder.build();

      List<Map<Integer, Float>> searchResult = null;
      try {
        searchResult = bruteforceIndex.search(query).getResults();
      } catch (Throwable t) {
        Utils.handleThrowable(t);
      }
      assert searchResult.size() == 1;
      result = searchResult.getFirst();
    }
    assert result != null;

    final var rawValues = flatVectorsReader.getFloatVectorValues(field);
    final Bits acceptedOrds = rawValues.getAcceptOrds(acceptDocs);
    final var ordToDocFunction = (IntToIntFunction) rawValues::ordToDoc;
    final var scoreCorrectionFunction = getScoreNormalizationFunc(fieldEntry.similarityFunction);

    for (var entry : result.entrySet()) {
      int ord = entry.getKey();
      float score = entry.getValue();
      if (acceptedOrds == null || acceptedOrds.get(ord)) {
        if (knnCollector.earlyTerminated()) {
          break;
        }
        assert ord >= 0 : "unexpected ord: " + ord;
        int doc = ordToDocFunction.apply(ord);
        float correctedScore = scoreCorrectionFunction.apply(score);
        knnCollector.incVisitedCount(1);
        knnCollector.collect(doc, correctedScore);
      }
    }
  }

  /**
   * Return the k nearest neighbor documents as determined by comparison of their vector values for this field, to the given vector.
   *
   * This is not supported.
   */
  @Override
  public void search(String field, byte[] target, KnnCollector knnCollector, Bits acceptDocs)
      throws IOException {
    throw new UnsupportedOperationException("byte vectors not supported");
  }

  /**
   * Holds the meta information for the field.
   */
  record FieldEntry(
      VectorEncoding vectorEncoding,
      VectorSimilarityFunction similarityFunction,
      int dims,
      int count,
      long cagraIndexOffset,
      long cagraIndexLength,
      long bruteForceIndexOffset,
      long bruteForceIndexLength) {

    /**
     * Returns an instance of FieldEntry.
     *
     * @param input instance of IndexInput
     * @param vectorEncoding The numeric datatype of the vector values
     * @param similarityFunction Vector similarity function; used in search to return top K most similar vectors to a target vector
     * @return an instance of FieldEntry
     * @throws IOException I/O Exceptions
     */
    static FieldEntry readEntry(
        IndexInput input,
        VectorEncoding vectorEncoding,
        VectorSimilarityFunction similarityFunction)
        throws IOException {
      var dims = input.readInt();
      var count = input.readInt();
      var cagraIndexOffset = input.readVLong();
      var cagraIndexLength = input.readVLong();
      var bruteForceIndexOffset = input.readVLong();
      var bruteForceIndexLength = input.readVLong();
      return new FieldEntry(
          vectorEncoding,
          similarityFunction,
          dims,
          count,
          cagraIndexOffset,
          cagraIndexLength,
          bruteForceIndexOffset,
          bruteForceIndexLength);
    }
  }

  /**
   * Checks the version and throws CorruptIndexException on mismatch.
   *
   * @param versionMeta
   * @param versionVectorData
   * @param in
   * @throws CorruptIndexException
   */
  static void checkVersion(int versionMeta, int versionVectorData, IndexInput in)
      throws CorruptIndexException {
    if (versionMeta != versionVectorData) {
      throw new CorruptIndexException(
          "Format versions mismatch: meta="
              + versionMeta
              + ", "
              + CUVS_META_CODEC_NAME
              + "="
              + versionVectorData,
          in);
    }
  }

  /**
   * Gets the instance of FieldInfos.
   *
   * @return the instance of FieldInfos
   */
  public FieldInfos getFieldInfos() {
    return fieldInfos;
  }

  /**
   * Gets the map of {@link GPUIndex} objects.
   *
   * @return the map of gpu index objects
   */
  public IntObjectHashMap<GPUIndex> getCuvsIndexes() {
    return cuvsIndices;
  }

  /**
   * Gets the map of FieldEntry objects that hold the meta information for the field.
   *
   * @return the map of FieldEntry objects
   */
  public IntObjectHashMap<FieldEntry> getFieldEntries() {
    return fields;
  }
}

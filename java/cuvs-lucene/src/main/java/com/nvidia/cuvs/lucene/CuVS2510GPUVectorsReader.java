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
import static com.nvidia.cuvs.lucene.CuVS2510GPUVectorsFormat.VERSION_START;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.closeCuVSResourcesInstance;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.getCuVSResourcesInstance;

import com.nvidia.cuvs.BruteForceIndex;
import com.nvidia.cuvs.BruteForceQuery;
import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraQuery;
import com.nvidia.cuvs.CagraSearchParams;
import com.nvidia.cuvs.CuVSMatrix;
import java.io.IOException;
import java.util.BitSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
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
import org.apache.lucene.store.IOContext.Context;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.ReadAdvice;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.hnsw.IntToIntFunction;

/**
 * KnnVectorsReader instance associated with cuVS format for reading vectors from an index.
 *
 * @since 25.10
 */
public class CuVS2510GPUVectorsReader extends KnnVectorsReader {

  private static final LuceneProvider LUCENE_PROVIDER;
  private static final List<VectorSimilarityFunction> VECTOR_SIMILARITY_FUNCTIONS;

  private final FlatVectorsReader flatVectorsReader;
  private final FieldInfos fieldInfos;
  private final IntObjectHashMap<FieldEntry> fields;
  private final IntObjectHashMap<GPUIndex> cuvsIndices;
  private final IndexInput cuvsIndexInput;

  static {
    try {
      LUCENE_PROVIDER = LuceneProvider.getInstance("99");
      VECTOR_SIMILARITY_FUNCTIONS = LUCENE_PROVIDER.getSimilarityFunctions();
    } catch (Exception e) {
      throw new ExceptionInInitializerError(e.getMessage());
    }
  }

  /**
   * Initializes the {@link CuVS2510GPUVectorsReader}, checks and loads the index.
   *
   * @param state instance of the SegmentReadState
   * @param flatReader instance of the FlatVectorsReader
   *
   * @throws IOException I/O exception
   */
  public CuVS2510GPUVectorsReader(SegmentReadState state, FlatVectorsReader flatReader)
      throws IOException {
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
      /*
       * Only load indexes on the GPU when this reader is opening for searches.
       * Do not load indexes on the GPU when this reader is opening during merge calls.
       * With this approach we reduce device memory usage by approximately 50% during merges.
       */
      if (state.context.context().equals(Context.MERGE)) {
        cuvsIndices = null;
      } else {
        cuvsIndices = loadCuVSIndices();
      }
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
   * @param meta instance of the ChecksumIndexInput
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

  /**
   * Checks the distance function validity and returns it.
   *
   * @param input instance of DataInput
   * @return an instance of VectorSimilarityFunction
   * @throws IOException
   */
  static VectorSimilarityFunction readSimilarityFunction(DataInput input) throws IOException {
    int i = input.readInt();
    if (i < 0 || i >= VECTOR_SIMILARITY_FUNCTIONS.size()) {
      throw new IllegalArgumentException("invalid distance function: " + i);
    }
    return VECTOR_SIMILARITY_FUNCTIONS.get(i);
  }

  /**
   * Reads the vector encoding (The numeric data type of the vector values) from the DataInput.
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
    for (var field : fields) {
      var fieldEntry = field.value;
      int fieldNumber = field.key;
      var cuvsIndex = loadCuVSIndex(fieldEntry);
      indices.put(fieldNumber, cuvsIndex);
    }
    return indices;
  }

  /**
   * Loads the CAGRA and brute force index (if exists) onto the GPU.
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
          cagraIndex = CagraIndex.newBuilder(getCuVSResourcesInstance()).from(in).build();
        }
      }
      len = fieldEntry.bruteForceIndexLength();
      if (len > 0) {
        long off = fieldEntry.bruteForceIndexOffset();
        try (var slice = cuvsIndexInput.slice("bf index", off, len);
            var in = new IndexInputInputStream(slice)) {
          bruteForceIndex = BruteForceIndex.newBuilder(getCuVSResourcesInstance()).from(in).build();
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
    var closeableStream = Stream.of(flatVectorsReader, cuvsIndexInput);
    IOUtils.close(closeableStream::iterator);
    if (cuvsIndices != null) {
      var indexClosableStream = stream(cuvsIndices.values().iterator()).map(cursor -> cursor.value);
      IOUtils.close(indexClosableStream::iterator);
    }
    closeCuVSResourcesInstance();
  }

  static <T> Stream<T> stream(Iterator<T> iterator) {
    return StreamSupport.stream(((Iterable<T>) () -> iterator).spliterator(), false);
  }

  /**
   * Checks consistency of this reader.
   */
  @Override
  public void checkIntegrity() throws IOException {
    flatVectorsReader.checkIntegrity();
    CodecUtil.checksumEntireFile(cuvsIndexInput);
  }

  /**
   * Returns the FloatVectorValues for the given field.
   */
  @Override
  public FloatVectorValues getFloatVectorValues(String field) throws IOException {
    return flatVectorsReader.getFloatVectorValues(field);
  }

  /**
   * Returns the ByteVectorValues for the given field.
   *
   * This is not supported.
   */
  @Override
  public ByteVectorValues getByteVectorValues(String field) {
    throw new UnsupportedOperationException("byte vectors are not currently supported");
  }

  /** Native float to float function */
  private interface FloatToFloatFunction {
    float apply(float v);
  }

  /**
   * Get the score normalization function.
   *
   * @param sim instance of VectorSimilarityFunction
   * @return an instance of the FloatToFloatFunction
   */
  private static FloatToFloatFunction getScoreNormalizationFunc(VectorSimilarityFunction sim) {
    // TODO: check for different similarities
    return score -> (1f / (1f + score));
  }

  /**
   * Returns the k nearest neighbor documents using cuVS's CAGRA or brute force algorithm for this field, to the given vector.
   */
  @Override
  public void search(String field, float[] target, KnnCollector knnCollector, Bits acceptDocs)
      throws IOException {
    var fieldEntry = getFieldEntry(field, VectorEncoding.FLOAT32);
    if (fieldEntry.count() == 0 || knnCollector.k() == 0) {
      return;
    }

    var fieldNumber = fieldInfos.fieldInfo(field).number;
    GPUIndex cuvsIndex = cuvsIndices != null ? cuvsIndices.get(fieldNumber) : null;
    if (cuvsIndex == null) {
      throw new IllegalStateException("Index not found for field:" + field);
    }

    final FloatVectorValues rawValues = flatVectorsReader.getFloatVectorValues(field);
    final Bits acceptedOrds = rawValues.getAcceptOrds(acceptDocs);
    BitSet[] mask = null;
    int maskLength = 0;
    int topK = knnCollector.k();

    if (acceptDocs != null) {
      mask = new BitSet[1]; // As there is only one query "target"
      mask[0] = new BitSet(acceptedOrds.length());
      /*
       * Need to find if there is a better alternative for below loop
       * in subsequent code improvement iterations. This is needed as
       * there is a difference between Lucene's Bits "acceptDocs" and
       * what our API accepts.
       */
      for (int i = 0; i < acceptedOrds.length(); i++) {
        if (acceptedOrds.get(i)) {
          mask[0].set(i);
        }
      }
      topK = Math.min(knnCollector.k() + 10, mask[0].cardinality());
      maskLength = mask[0].length();
    }

    try {
      List<Map<Integer, Float>> searchResult = null;
      if (knnCollector.k() <= 1024 && cuvsIndex.getCagraIndex() != null) {
        CagraSearchParams searchParams;
        if (knnCollector instanceof GPUPerLeafCuVSKnnCollector) {
          GPUPerLeafCuVSKnnCollector collector = (GPUPerLeafCuVSKnnCollector) knnCollector;
          searchParams =
              new CagraSearchParams.Builder()
                  .withItopkSize(Math.max(collector.getiTopK(), topK))
                  .withSearchWidth(collector.getSearchWidth())
                  .build();
        } else {
          // Setting itopK as topK because in any case iTopK should be ATLEAST equal to topK
          searchParams = new CagraSearchParams.Builder().withItopkSize(topK).build();
        }
        CagraIndex cagraIndex = cuvsIndex.getCagraIndex();
        assert cagraIndex != null;
        CagraQuery query = null;

        CuVSMatrix.Builder<?> builder =
            CuVSMatrix.deviceBuilder(
                getCuVSResourcesInstance(), 1, target.length, CuVSMatrix.DataType.FLOAT);
        builder.addVector(target);
        CuVSMatrix queryVector = builder.build();

        if (acceptDocs != null) {
          query =
              new CagraQuery.Builder(getCuVSResourcesInstance())
                  .withTopK(topK)
                  .withSearchParams(searchParams)
                  .withQueryVectors(queryVector)
                  .withPrefilter(mask[0], maskLength)
                  .build();
        } else {
          query =
              new CagraQuery.Builder(getCuVSResourcesInstance())
                  .withTopK(topK)
                  .withSearchParams(searchParams)
                  .withQueryVectors(queryVector)
                  .build();
        }
        searchResult = cagraIndex.search(query).getResults();
      } else {
        BruteForceIndex bruteforceIndex = cuvsIndex.getBruteforceIndex();
        assert bruteforceIndex != null;
        BruteForceQuery query = null;
        float[][] queryVector = new float[][] {target};
        if (acceptDocs != null) {
          query =
              new BruteForceQuery.Builder(getCuVSResourcesInstance())
                  .withQueryVectors(queryVector)
                  .withPrefilters(mask, maskLength)
                  .withTopK(topK)
                  .build();
        } else {
          query =
              new BruteForceQuery.Builder(getCuVSResourcesInstance())
                  .withQueryVectors(queryVector)
                  .withTopK(topK)
                  .build();
        }
        searchResult = bruteforceIndex.search(query).getResults();
      }

      // List expected to have only one entry because of single query "target".
      assert searchResult.size() == 1;
      final IntToIntFunction ordToDocFunction = (IntToIntFunction) rawValues::ordToDoc;
      final FloatToFloatFunction scoreCorrectionFunction =
          getScoreNormalizationFunc(fieldEntry.similarityFunction);

      for (Entry<Integer, Float> entry : searchResult.getFirst().entrySet()) {
        int ord = entry.getKey();
        float score = entry.getValue();
        if (knnCollector.earlyTerminated()) {
          break;
        }
        if (ord < 0) {
          continue;
        }
        float correctedScore = scoreCorrectionFunction.apply(score);
        int doc = ordToDocFunction.apply(ord);
        knnCollector.incVisitedCount(1);
        knnCollector.collect(doc, correctedScore);
      }
    } catch (Throwable t) {
      Utils.handleThrowable(t);
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
    throw new UnsupportedOperationException("Byte vectors are not currently supported");
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
     * @param vectorEncoding The numeric data type of the vector values
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
  private static void checkVersion(int versionMeta, int versionVectorData, IndexInput in)
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
   * @return the map of GPU index objects
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

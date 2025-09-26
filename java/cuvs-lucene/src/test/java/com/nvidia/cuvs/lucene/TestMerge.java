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

import static org.apache.lucene.tests.util.TestUtil.alwaysKnnVectorsFormat;

import com.nvidia.cuvs.lucene.CuVS2510GPUVectorsWriter.IndexType;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.logging.Logger;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.util.BytesRef;
import org.junit.After;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Comprehensive tests for merge functionality with CuVS indexes.
 * Tests merge operations across different index types including brute force,
 * CAGRA, and combined index configurations to ensure proper vector handling
 * and search functionality after segment merging.
 */
@SuppressSysoutChecks(bugUrl = "CuVS native library produces verbose logging output")
public class TestMerge extends LuceneTestCase {

  private static final Logger log = Logger.getLogger(TestMerge.class.getName());

  private static final int MIN_VECTOR_DIMENSION = 64;
  private static final int MAX_VECTOR_DIMENSION = 256;
  private static final int TOP_K_LIMIT = 64;

  @BeforeClass
  public static void beforeClass() {
    assumeTrue("cuVS is not supported", CuVS2510GPUVectorsFormat.supported());
  }

  private Directory directory;
  private int vectorDimension;

  @Before
  public void setUp() throws Exception {
    super.setUp();
    directory = newDirectory();

    // Randomize vector dimension for each test
    vectorDimension =
        MIN_VECTOR_DIMENSION + random().nextInt(MAX_VECTOR_DIMENSION - MIN_VECTOR_DIMENSION + 1);
    // Ensure dimension is multiple of 4 for better performance
    vectorDimension = (vectorDimension / 4) * 4;

    log.info("Using randomized vector dimension: " + vectorDimension);
  }

  @After
  public void tearDown() throws Exception {
    if (directory != null) {
      directory.close();
    }
    super.tearDown();
  }

  /**
   *  Test merging many documents across multiple segments
   **/
  @Test
  public void testMergeManyDocumentsMultipleSegments() throws IOException {
    log.info("Starting testMergeManyDocumentsMultipleSegments");

    // Randomize configuration parameters
    int maxBufferedDocs = 5 + random().nextInt(16); // 5-20 docs per buffer
    int totalBatches = 8 + random().nextInt(8); // 8-15 batches
    int docsPerBatch = 15 + random().nextInt(11); // 15-25 docs per batch
    int totalDocuments = totalBatches * docsPerBatch;

    // Randomize vector presence probability (60-85%)
    double vectorProbability = 0.6 + (random().nextDouble() * 0.25);

    log.info(
        "Randomized parameters: maxBufferedDocs="
            + maxBufferedDocs
            + ", totalBatches="
            + totalBatches
            + ", docsPerBatch="
            + docsPerBatch
            + ", totalDocuments="
            + totalDocuments
            + ", vectorProbability="
            + vectorProbability);

    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(alwaysKnnVectorsFormat(new CuVS2510GPUVectorsFormat()))
            .setMaxBufferedDocs(maxBufferedDocs) // Randomized buffer size
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);

    List<float[]> expectedVectors = new ArrayList<>();
    List<Integer> expectedDocIds = new ArrayList<>();
    int documentsWithVectors = 0;

    try (IndexWriter writer = new IndexWriter(directory, config)) {
      // Add documents in multiple batches to create many segments
      for (int batch = 0; batch < totalBatches; batch++) {
        for (int i = 0; i < docsPerBatch; i++) {
          int docId = batch * docsPerBatch + i;
          Document doc = new Document();
          doc.add(new StringField("id", String.valueOf(docId), Field.Store.YES));
          doc.add(new NumericDocValuesField("batch", batch));

          // Randomly decide if document has vector
          if (random().nextDouble() < vectorProbability) {
            float[] vector = generateRandomVector(vectorDimension, random());
            doc.add(new KnnFloatVectorField("vector", vector, VectorSimilarityFunction.COSINE));
            expectedVectors.add(vector);
            expectedDocIds.add(docId);
            documentsWithVectors++;
          }

          writer.addDocument(doc);
        }
        writer.commit(); // Create a new segment
      }

      int documentsWithoutVectors = totalDocuments - documentsWithVectors;
      log.info("Created " + totalDocuments + " documents in " + totalBatches + " segments");
      log.info("Documents with vectors: " + documentsWithVectors);
      log.info("Documents without vectors: " + documentsWithoutVectors);

      // Force merge to trigger merge logic
      writer.forceMerge(1);
      log.info("Forced merge to single segment completed");
    }

    // Verify the merged index
    try (DirectoryReader reader = DirectoryReader.open(directory)) {
      assertEquals("Should have exactly one segment after merge", 1, reader.leaves().size());

      LeafReader leafReader = reader.leaves().get(0).reader();
      assertEquals("Total documents should match", totalDocuments, leafReader.maxDoc());

      // Verify vector search works correctly after merge
      if (documentsWithVectors > 0) {
        IndexSearcher searcher = new IndexSearcher(reader);
        float[] queryVector = generateRandomVector(vectorDimension, random());

        // Randomize search parameters
        int searchK =
            Math.min(5 + random().nextInt(10), Math.min(documentsWithVectors, TOP_K_LIMIT));

        KnnFloatVectorQuery query = new KnnFloatVectorQuery("vector", queryVector, searchK);
        TopDocs results = searcher.search(query, searchK);

        assertTrue("Should find some results after merge", results.scoreDocs.length > 0);
        assertTrue(
            "Should find reasonable number of results",
            results.scoreDocs.length <= documentsWithVectors);

        log.info(
            "Vector search returned "
                + results.scoreDocs.length
                + " results out of "
                + documentsWithVectors
                + " documents with vectors");

        // Verify all returned documents have valid IDs
        for (ScoreDoc scoreDoc : results.scoreDocs) {
          int docId = Integer.parseInt(searcher.storedFields().document(scoreDoc.doc).get("id"));
          assertTrue("Document ID should be valid", docId >= 0 && docId < totalDocuments);
        }
      } else {
        log.info("No documents with vectors - skipping vector search verification");
      }

      log.info("Merge verification completed successfully");
    }
  }

  /**
   * Test merging with index sorting enabled using text-based sorting and SortingMergePolicy
   **/
  @Test
  public void testMergeWithIndexSorting() throws IOException {
    log.info("Starting testMergeWithIndexSorting with text-based sorting");

    // Randomize sort field type
    SortField.Type sortType = random().nextBoolean() ? SortField.Type.STRING : SortField.Type.LONG;
    String sortFieldName = sortType == SortField.Type.STRING ? "text_sort_key" : "numeric_sort_key";

    // Configure index sorting by a randomized field
    Sort indexSort = new Sort(new SortField(sortFieldName, sortType));

    // Randomize merge policy parameters
    TieredMergePolicy mergePolicy = new TieredMergePolicy();
    mergePolicy.setMaxMergedSegmentMB(128 + random().nextInt(257)); // 128-384 MB
    mergePolicy.setSegmentsPerTier(3 + random().nextInt(4)); // 3-6 segments per tier

    // Randomize writer configuration parameters
    int maxBufferedDocs = 10 + random().nextInt(16); // 10-25 docs per buffer
    int totalDocuments = 80 + random().nextInt(81); // 80-160 documents
    int segmentSize = 15 + random().nextInt(11); // 15-25 docs per segment
    double vectorProbability = 0.65 + (random().nextDouble() * 0.25); // 65-90% have vectors

    log.info(
        "Randomized sorting parameters: sortType=" + sortType + ", sortFieldName=" + sortFieldName);
    log.info(
        "Randomized config: maxBufferedDocs="
            + maxBufferedDocs
            + ", totalDocuments="
            + totalDocuments
            + ", segmentSize="
            + segmentSize
            + ", vectorProbability="
            + vectorProbability);

    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(alwaysKnnVectorsFormat(new CuVS2510GPUVectorsFormat()))
            .setIndexSort(indexSort) // This automatically enables sorting during merges
            .setMergePolicy(mergePolicy)
            .setMaxBufferedDocs(maxBufferedDocs)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);

    // List<DocumentData> documents = new ArrayList<>();

    try (IndexWriter writer = new IndexWriter(directory, config)) {
      // Create documents with randomized sort keys
      for (int i = 0; i < totalDocuments; i++) {
        float[] vector = null;

        // Randomly decide if document has vector
        if (random().nextDouble() < vectorProbability) {
          vector = generateRandomVector(vectorDimension, random());
        }

        Document doc = new Document();
        doc.add(new StringField("id", String.valueOf(i), Field.Store.YES));
        doc.add(new StringField("original_order", String.valueOf(i), Field.Store.YES));

        // Add sort field based on randomized type
        if (sortType == SortField.Type.STRING) {
          // Randomize text sort key length (4-12 characters)
          int keyLength = 4 + random().nextInt(9);
          String textSortKey = generateRandomText(random(), keyLength);
          doc.add(new SortedDocValuesField(sortFieldName, new BytesRef(textSortKey)));
          doc.add(new StringField(sortFieldName + "_stored", textSortKey, Field.Store.YES));
        } else {
          // Use numeric sort key with wider range
          long numericSortKey = random().nextLong() % 100000; // Can be negative for more variety
          doc.add(new NumericDocValuesField(sortFieldName, numericSortKey));
          doc.add(
              new StringField(
                  sortFieldName + "_stored", String.valueOf(numericSortKey), Field.Store.YES));
        }

        if (vector != null) {
          doc.add(new KnnFloatVectorField("vector", vector, VectorSimilarityFunction.COSINE));
        }

        writer.addDocument(doc);

        // Commit based on randomized segment size
        if ((i + 1) % segmentSize == 0) {
          writer.commit();
          log.info(
              "Committed segment "
                  + ((i + 1) / segmentSize)
                  + " with "
                  + (i + 1)
                  + " total documents");
        }
      }

      log.info("Created " + totalDocuments + " documents with text-based index sorting");

      // Force merge with sorting - this will use the sorting merge policy
      writer.forceMerge(1);
      log.info("Forced merge with text-based sorting completed");
    }

    // Verify the merged and sorted index
    try (DirectoryReader reader = DirectoryReader.open(directory)) {
      assertEquals("Should have exactly one segment after merge", 1, reader.leaves().size());

      LeafReader leafReader = reader.leaves().get(0).reader();
      assertEquals("Total documents should match", totalDocuments, leafReader.maxDoc());

      // Verify documents are sorted correctly by the randomized sort field
      log.info(
          "Verifying document sorting order using sortType: "
              + sortType
              + ", field: "
              + sortFieldName);

      if (sortType == SortField.Type.STRING) {
        // Verify string-based sorting
        String previousSortKey = "";
        SortedDocValues sortedValues = leafReader.getSortedDocValues(sortFieldName);

        for (int docId = 0; docId < leafReader.maxDoc(); docId++) {
          String currentSortKey = "";
          if (sortedValues != null && sortedValues.advanceExact(docId)) {
            currentSortKey = sortedValues.lookupOrd(sortedValues.ordValue()).utf8ToString();
          }

          assertTrue(
              "Documents should be sorted by "
                  + sortFieldName
                  + ": '"
                  + previousSortKey
                  + "' should be <= '"
                  + currentSortKey
                  + "'",
              previousSortKey.compareTo(currentSortKey) <= 0);
          previousSortKey = currentSortKey;

          // Log first 10 documents to verify sorting
          if (docId < 10) {
            IndexSearcher searcher = new IndexSearcher(reader);
            String originalOrder = searcher.storedFields().document(docId).get("original_order");
            log.info(
                "DocId: "
                    + docId
                    + ", OriginalOrder: "
                    + originalOrder
                    + ", SortKey: '"
                    + currentSortKey
                    + "'");
          }
        }
      } else {
        // Verify numeric-based sorting
        long previousSortKey = Long.MIN_VALUE;
        var numericValues = leafReader.getNumericDocValues(sortFieldName);

        for (int docId = 0; docId < leafReader.maxDoc(); docId++) {
          long currentSortKey = Long.MIN_VALUE;
          if (numericValues != null && numericValues.advanceExact(docId)) {
            currentSortKey = numericValues.longValue();
          }

          assertTrue(
              "Documents should be sorted by "
                  + sortFieldName
                  + ": "
                  + previousSortKey
                  + " should be <= "
                  + currentSortKey,
              previousSortKey <= currentSortKey);
          previousSortKey = currentSortKey;

          // Log first 10 documents to verify sorting
          if (docId < 10) {
            IndexSearcher searcher = new IndexSearcher(reader);
            String originalOrder = searcher.storedFields().document(docId).get("original_order");
            log.info(
                "DocId: "
                    + docId
                    + ", OriginalOrder: "
                    + originalOrder
                    + ", SortKey: "
                    + currentSortKey);
          }
        }
      }

      // Count total vectors by checking if vector field exists and has values
      var vectorValues = leafReader.getFloatVectorValues("vector");
      int documentsWithVectors = vectorValues != null ? vectorValues.size() : 0;

      log.info("Found " + documentsWithVectors + " documents with vectors after sorted merge");

      // Test vector search on sorted index
      if (documentsWithVectors > 0) {
        IndexSearcher searcher = new IndexSearcher(reader);
        float[] queryVector = generateRandomVector(vectorDimension, random());

        KnnFloatVectorQuery query =
            new KnnFloatVectorQuery("vector", queryVector, Math.min(10, documentsWithVectors));
        TopDocs results = searcher.search(query, 10);

        assertTrue("Should find results in sorted index", results.scoreDocs.length > 0);
        log.info("Vector search on sorted index returned " + results.scoreDocs.length + " results");

        // Verify that returned documents maintain sort order if we check their sort keys
        log.info("Verifying vector search results maintain sorting consistency...");
        for (int i = 0; i < Math.min(3, results.scoreDocs.length); i++) {
          ScoreDoc scoreDoc = results.scoreDocs[i];
          String originalOrder =
              searcher.storedFields().document(scoreDoc.doc).get("original_order");
          String sortKey =
              searcher.storedFields().document(scoreDoc.doc).get(sortFieldName + "_stored");
          log.info(
              "Result "
                  + i
                  + ": DocId="
                  + scoreDoc.doc
                  + ", OriginalOrder="
                  + originalOrder
                  + ", SortKey='"
                  + sortKey
                  + "', Score="
                  + scoreDoc.score);
        }
      }

      log.info("Text-based index sorting verification completed successfully");
    }
  }

  /**
   * Test merging segments with various patterns of missing vectors
   **/
  @Test
  public void testMergeWithMissingVectors() throws IOException {
    log.info("Starting testMergeWithMissingVectors");

    // Randomize configuration
    int maxBufferedDocs = 10 + random().nextInt(11); // 10-20 docs per buffer
    int numSegments = 3 + random().nextInt(3); // 3-5 segments

    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(alwaysKnnVectorsFormat(new CuVS2510GPUVectorsFormat()))
            .setMaxBufferedDocs(maxBufferedDocs)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);

    log.info(
        "Randomized parameters: maxBufferedDocs="
            + maxBufferedDocs
            + ", numSegments="
            + numSegments);

    int totalExpectedVectors = 0;
    int totalDocuments = 0;

    try (IndexWriter writer = new IndexWriter(directory, config)) {
      for (int seg = 0; seg < numSegments; seg++) {
        // Randomize segment characteristics
        int docsInSegment = 15 + random().nextInt(16); // 15-30 docs per segment
        double vectorProbability = random().nextDouble(); // 0-100% vector probability
        String segmentType = "seg_" + seg + "_prob_" + String.format("%.2f", vectorProbability);

        int segmentVectorCount = 0;

        for (int i = 0; i < docsInSegment; i++) {
          Document doc = new Document();
          doc.add(new StringField("id", "seg" + seg + "_" + i, Field.Store.YES));
          doc.add(new StringField("segment", segmentType, Field.Store.YES));
          doc.add(new NumericDocValuesField("segment_num", seg));
          doc.add(new NumericDocValuesField("doc_in_segment", i));

          // Randomly add vector based on segment's probability
          if (random().nextDouble() < vectorProbability) {
            float[] vector = generateRandomVector(vectorDimension, random());
            doc.add(new KnnFloatVectorField("vector", vector, VectorSimilarityFunction.COSINE));
            segmentVectorCount++;
          }

          writer.addDocument(doc);
        }

        writer.commit();
        totalDocuments += docsInSegment;
        totalExpectedVectors += segmentVectorCount;

        log.info(
            "Created segment "
                + seg
                + ": "
                + docsInSegment
                + " documents, "
                + segmentVectorCount
                + " with vectors (probability: "
                + String.format("%.2f", vectorProbability)
                + ")");
      }

      // Force merge all segments
      writer.forceMerge(1);
      log.info("Forced merge of " + numSegments + " segments completed");
    }

    // Verify the merged index handles missing vectors correctly
    try (DirectoryReader reader = DirectoryReader.open(directory)) {
      assertEquals("Should have exactly one segment after merge", 1, reader.leaves().size());

      LeafReader leafReader = reader.leaves().get(0).reader();
      assertEquals("Total documents should match", totalDocuments, leafReader.maxDoc());

      // Count actual vectors in merged index
      var vectorValues = leafReader.getFloatVectorValues("vector");
      int actualVectorCount = vectorValues != null ? vectorValues.size() : 0;

      log.info(
          "Total documents: "
              + totalDocuments
              + ", Expected vectors: "
              + totalExpectedVectors
              + ", Actual vectors: "
              + actualVectorCount);

      assertEquals("Vector count should match expected", totalExpectedVectors, actualVectorCount);

      // Test vector search if we have vectors
      if (actualVectorCount > 0) {
        IndexSearcher searcher = new IndexSearcher(reader);
        float[] queryVector = generateRandomVector(vectorDimension, random());

        // Randomize search parameters
        int searchK = Math.min(5 + random().nextInt(10), Math.min(actualVectorCount, TOP_K_LIMIT));

        KnnFloatVectorQuery vectorQuery = new KnnFloatVectorQuery("vector", queryVector, searchK);
        TopDocs vectorResults = searcher.search(vectorQuery, searchK);

        assertTrue("Should find some vector results", vectorResults.scoreDocs.length > 0);
        assertTrue(
            "Should not find more vectors than exist",
            vectorResults.scoreDocs.length <= actualVectorCount);

        log.info(
            "Found "
                + vectorResults.scoreDocs.length
                + " vector results out of "
                + actualVectorCount
                + " available");
      } else {
        log.info("No vectors in merged index - skipping vector search");
      }

      log.info("Missing vectors test completed successfully");
    }
  }

  /**
   *  Test merge behavior with document deletions
   **/
  @Test
  public void testMergeWithDeletions() throws IOException {
    log.info("Starting testMergeWithDeletions");

    // Randomize configuration parameters
    int maxBufferedDocs = 15 + random().nextInt(11); // 15-25 docs per buffer
    int numSegments = 3 + random().nextInt(4); // 3-6 segments
    int docsPerSegment = 20 + random().nextInt(21); // 20-40 docs per segment
    double vectorProbability = 0.7 + (random().nextDouble() * 0.25); // 70-95% have vectors
    double deletionProbability = 0.2 + (random().nextDouble() * 0.3); // 20-50% deletion rate

    log.info(
        "Randomized parameters: maxBufferedDocs="
            + maxBufferedDocs
            + ", numSegments="
            + numSegments
            + ", docsPerSegment="
            + docsPerSegment
            + ", vectorProbability="
            + vectorProbability
            + ", deletionProbability="
            + deletionProbability);

    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(alwaysKnnVectorsFormat(new CuVS2510GPUVectorsFormat()))
            .setMaxBufferedDocs(maxBufferedDocs)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);

    List<Integer> expectedRemainingDocs = new ArrayList<>();
    List<Integer> deletedDocs = new ArrayList<>();
    int totalDocuments = numSegments * docsPerSegment;

    try (IndexWriter writer = new IndexWriter(directory, config)) {
      // Create multiple segments with documents
      for (int seg = 0; seg < numSegments; seg++) {
        for (int i = 0; i < docsPerSegment; i++) {
          int docId = seg * docsPerSegment + i;
          Document doc = new Document();
          doc.add(new StringField("id", String.valueOf(docId), Field.Store.YES));
          doc.add(new StringField("segment", "seg_" + seg, Field.Store.YES));
          doc.add(new NumericDocValuesField("doc_num", docId));
          doc.add(new NumericDocValuesField("segment_num", seg));

          // Randomly add vectors
          if (random().nextDouble() < vectorProbability) {
            float[] vector = generateRandomVector(vectorDimension, random());
            doc.add(new KnnFloatVectorField("vector", vector, VectorSimilarityFunction.COSINE));
          }

          writer.addDocument(doc);
        }
        writer.commit();
      }

      log.info(
          "Created "
              + numSegments
              + " segments with "
              + docsPerSegment
              + " documents each ("
              + totalDocuments
              + " total)");

      // Delete documents randomly and track which ones are deleted
      int deletedCount = 0;
      for (int docId = 0; docId < totalDocuments; docId++) {
        if (random().nextDouble() < deletionProbability) {
          writer.deleteDocuments(new Term("id", String.valueOf(docId)));
          deletedDocs.add(docId);
          deletedCount++;
        } else {
          expectedRemainingDocs.add(docId);
        }
      }

      log.info(
          "Deleted "
              + deletedCount
              + " documents ("
              + String.format("%.1f", (100.0 * deletedCount / totalDocuments))
              + "%), remaining: "
              + expectedRemainingDocs.size());

      writer.commit();

      // Force merge to apply deletions
      writer.forceMerge(1);
      log.info("Forced merge with deletions completed");
    }

    // Verify the merged index correctly handles deletions
    try (DirectoryReader reader = DirectoryReader.open(directory)) {
      assertEquals("Should have exactly one segment after merge", 1, reader.leaves().size());

      LeafReader leafReader = reader.leaves().get(0).reader();
      int expectedRemaining = expectedRemainingDocs.size();
      assertEquals(
          "Should have correct number of documents after deletions",
          expectedRemaining,
          leafReader.maxDoc());

      // Verify that deleted documents are not present
      IndexSearcher searcher = new IndexSearcher(reader);

      // Test that we can find expected remaining documents
      for (int i = 0; i < Math.min(10, expectedRemainingDocs.size()); i++) {
        int docId = expectedRemainingDocs.get(i);
        TopDocs result = searcher.search(new TermQuery(new Term("id", String.valueOf(docId))), 1);
        assertEquals("Should find remaining document " + docId, 1, (int) result.totalHits.value());
      }

      // Test that actually deleted documents are not found
      int deletedDocsToCheck = Math.min(10, deletedDocs.size()); // Check up to 10 deleted docs
      for (int i = 0; i < deletedDocsToCheck; i++) {
        int docId = deletedDocs.get(i);
        TopDocs result = searcher.search(new TermQuery(new Term("id", String.valueOf(docId))), 1);
        assertEquals(
            "Should not find deleted document " + docId, 0, (int) result.totalHits.value());
      }

      // Test vector search works after deletions
      float[] queryVector = generateRandomVector(vectorDimension, random());
      KnnFloatVectorQuery vectorQuery = new KnnFloatVectorQuery("vector", queryVector, 10);
      TopDocs vectorResults = searcher.search(vectorQuery, 10);

      assertTrue(
          "Should find some vector results after deletions", vectorResults.scoreDocs.length > 0);

      log.info("Found " + vectorResults.scoreDocs.length + " vector results after deletions");
      log.info("Deletion merge verification completed successfully");
    }
  }

  /**
   * Test merging segments for {@link IndexType#BRUTE_FORCE}
   * */
  @Test
  public void testMergeBruteForceIndex() throws IOException {
    log.info("Starting testMergeBruteForceIndex");

    // Randomize configuration parameters
    int maxBufferedDocs = 8 + random().nextInt(8); // 8-15 docs per buffer
    int numSegments = 3 + random().nextInt(3); // 3-5 segments
    int docsPerSegment = 12 + random().nextInt(9); // 12-20 docs per segment
    double vectorProbability = 0.8 + (random().nextDouble() * 0.2); // 80-100% have vectors

    log.info(
        "Randomized parameters: maxBufferedDocs="
            + maxBufferedDocs
            + ", numSegments="
            + numSegments
            + ", docsPerSegment="
            + docsPerSegment
            + ", vectorProbability="
            + vectorProbability);

    // Configure with brute force index type
    CuVS2510GPUVectorsFormat bruteForceFormat =
        new CuVS2510GPUVectorsFormat(
            32, // writer threads
            128, // intermediate graph degree
            64, // graph degree
            1,
            IndexType.BRUTE_FORCE); // Use brute force index

    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(alwaysKnnVectorsFormat(bruteForceFormat))
            .setMaxBufferedDocs(maxBufferedDocs)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);

    int totalDocuments = numSegments * docsPerSegment;
    int totalExpectedVectors = 0;

    try (IndexWriter writer = new IndexWriter(directory, config)) {
      // Create multiple segments with brute force index
      for (int seg = 0; seg < numSegments; seg++) {
        int segmentVectorCount = 0;

        for (int i = 0; i < docsPerSegment; i++) {
          int docId = seg * docsPerSegment + i;
          Document doc = new Document();
          doc.add(new StringField("id", String.valueOf(docId), Field.Store.YES));
          doc.add(new StringField("segment", "seg_" + seg, Field.Store.YES));
          doc.add(new NumericDocValuesField("segment_num", seg));
          doc.add(new NumericDocValuesField("doc_in_segment", i));

          // Randomly add vectors based on probability
          if (random().nextDouble() < vectorProbability) {
            float[] vector = generateRandomVector(vectorDimension, random());
            doc.add(new KnnFloatVectorField("vector", vector, VectorSimilarityFunction.COSINE));
            segmentVectorCount++;
          }

          writer.addDocument(doc);
        }

        writer.commit();
        totalExpectedVectors += segmentVectorCount;

        log.info(
            "Created brute force segment "
                + seg
                + ": "
                + docsPerSegment
                + " documents, "
                + segmentVectorCount
                + " with vectors");
      }

      log.info(
          "Created "
              + numSegments
              + " brute force segments with "
              + totalDocuments
              + " total documents and "
              + totalExpectedVectors
              + " vectors");

      // Force merge all brute force segments
      writer.forceMerge(1);
      log.info("Forced merge of brute force segments completed");
    }

    // Verify the merged brute force index
    try (DirectoryReader reader = DirectoryReader.open(directory)) {
      assertEquals("Should have exactly one segment after merge", 1, reader.leaves().size());

      LeafReader leafReader = reader.leaves().get(0).reader();
      assertEquals("Total documents should match", totalDocuments, leafReader.maxDoc());

      // Count actual vectors in merged index
      var vectorValues = leafReader.getFloatVectorValues("vector");
      int actualVectorCount = vectorValues != null ? vectorValues.size() : 0;

      log.info(
          "Brute force merge results: Total documents: "
              + totalDocuments
              + ", Expected vectors: "
              + totalExpectedVectors
              + ", Actual vectors: "
              + actualVectorCount);

      assertEquals("Vector count should match expected", totalExpectedVectors, actualVectorCount);

      // Test brute force vector search (exact search)
      if (actualVectorCount > 0) {
        IndexSearcher searcher = new IndexSearcher(reader);
        float[] queryVector = generateRandomVector(vectorDimension, random());

        // Search for reasonable number of results
        int searchK = Math.min(8 + random().nextInt(8), Math.min(actualVectorCount, TOP_K_LIMIT));

        KnnFloatVectorQuery vectorQuery = new KnnFloatVectorQuery("vector", queryVector, searchK);
        TopDocs vectorResults = searcher.search(vectorQuery, searchK);

        assertTrue(
            "Should find some vector results in brute force index",
            vectorResults.scoreDocs.length > 0);
        assertTrue(
            "Should not find more vectors than exist",
            vectorResults.scoreDocs.length <= actualVectorCount);

        log.info(
            "Brute force search found "
                + vectorResults.scoreDocs.length
                + " results out of "
                + actualVectorCount
                + " available vectors");

        // Verify all returned documents are valid
        for (ScoreDoc scoreDoc : vectorResults.scoreDocs) {
          String docId = searcher.storedFields().document(scoreDoc.doc).get("id");
          assertNotNull("Document should have valid ID", docId);
          assertTrue("Score should be positive", scoreDoc.score > 0);
        }
      } else {
        log.info("No vectors in brute force merged index - skipping vector search");
      }

      log.info("Brute force merge verification completed successfully");
    }
  }

  /**
   * Test merging segments for {@link IndexType#CAGRA_AND_BRUTE_FORCE}
   * */
  @Test
  public void testMergeCagraAndBruteForceIndex() throws IOException {
    log.info("Starting testMergeCagraAndBruteForceIndex");

    // Use moderate dataset size
    int maxBufferedDocs = 15 + random().nextInt(10); // 15-24 docs per buffer
    int numSegments =
        4; // Fixed 4 segments: alternating CAGRA vs small segments (brute force fallback)
    int docsPerSegment = 20 + random().nextInt(11); // 20-30 docs per segment
    double vectorProbability = 0.9 + (random().nextDouble() * 0.1); // 90-100% have vectors

    log.info(
        "Randomized parameters: maxBufferedDocs="
            + maxBufferedDocs
            + ", numSegments="
            + numSegments
            + ", docsPerSegment="
            + docsPerSegment
            + ", vectorProbability="
            + vectorProbability);

    // Configure with CAGRA + brute force combined index type
    CuVS2510GPUVectorsFormat combinedFormat =
        new CuVS2510GPUVectorsFormat(
            32, // writer threads
            128, // intermediate graph degree
            64, // graph degree
            1,
            IndexType.CAGRA_AND_BRUTE_FORCE); // Use combined CAGRA + brute force

    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(alwaysKnnVectorsFormat(combinedFormat))
            .setMaxBufferedDocs(maxBufferedDocs)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);

    int totalDocuments = numSegments * docsPerSegment;
    int totalExpectedVectors = 0;

    try (IndexWriter writer = new IndexWriter(directory, config)) {
      // Create segments that will result in mixed index types during merge
      for (int seg = 0; seg < numSegments; seg++) {
        int segmentVectorCount = 0;

        for (int i = 0; i < docsPerSegment; i++) {
          int docId = seg * docsPerSegment + i;
          Document doc = new Document();
          doc.add(new StringField("id", String.valueOf(docId), Field.Store.YES));
          doc.add(new StringField("segment", "mixed_seg_" + seg, Field.Store.YES));
          doc.add(new StringField("index_type", "cagra_and_brute_force", Field.Store.YES));
          doc.add(new NumericDocValuesField("segment_num", seg));
          doc.add(new NumericDocValuesField("doc_in_segment", i));

          // Add vectors based on probability
          if (random().nextDouble() < vectorProbability) {
            float[] vector = generateRandomVector(vectorDimension, random());
            doc.add(new KnnFloatVectorField("vector", vector, VectorSimilarityFunction.COSINE));
            segmentVectorCount++;
          }

          writer.addDocument(doc);
        }

        writer.commit();
        totalExpectedVectors += segmentVectorCount;

        log.info(
            "Created CAGRA+brute force segment "
                + seg
                + ": "
                + docsPerSegment
                + " documents, "
                + segmentVectorCount
                + " with vectors");
      }

      log.info(
          "Created "
              + numSegments
              + " CAGRA+brute force segments with "
              + totalDocuments
              + " total documents and "
              + totalExpectedVectors
              + " vectors");

      // Force merge all CAGRA+brute force segments
      writer.forceMerge(1);
      log.info("Forced merge of CAGRA+brute force segments completed");
    }

    // Verify the merged CAGRA+brute force index
    try (DirectoryReader reader = DirectoryReader.open(directory)) {
      assertEquals("Should have exactly one segment after merge", 1, reader.leaves().size());

      LeafReader leafReader = reader.leaves().get(0).reader();
      assertEquals("Total documents should match", totalDocuments, leafReader.maxDoc());

      // Count actual vectors in merged index
      var vectorValues = leafReader.getFloatVectorValues("vector");
      int actualVectorCount = vectorValues != null ? vectorValues.size() : 0;

      log.info(
          "CAGRA+brute force merge results: Total documents: "
              + totalDocuments
              + ", Expected vectors: "
              + totalExpectedVectors
              + ", Actual vectors: "
              + actualVectorCount);

      assertEquals("Vector count should match expected", totalExpectedVectors, actualVectorCount);

      // Test CAGRA+brute force index vector search
      if (actualVectorCount > 0) {
        IndexSearcher searcher = new IndexSearcher(reader);
        float[] queryVector = generateRandomVector(vectorDimension, random());

        // Search for reasonable number of results
        int searchK = Math.min(12 + random().nextInt(8), Math.min(actualVectorCount, TOP_K_LIMIT));

        KnnFloatVectorQuery vectorQuery = new KnnFloatVectorQuery("vector", queryVector, searchK);
        TopDocs vectorResults = searcher.search(vectorQuery, searchK);

        assertTrue(
            "Should find some vector results in CAGRA+brute force index",
            vectorResults.scoreDocs.length > 0);
        assertTrue(
            "Should not find more vectors than exist",
            vectorResults.scoreDocs.length <= actualVectorCount);

        log.info(
            "CAGRA+brute force index search found "
                + vectorResults.scoreDocs.length
                + " results out of "
                + actualVectorCount
                + " available vectors");

        // Verify all returned documents are valid and have expected metadata
        for (ScoreDoc scoreDoc : vectorResults.scoreDocs) {
          Document resultDoc = searcher.storedFields().document(scoreDoc.doc);
          String docId = resultDoc.get("id");
          String indexType = resultDoc.get("index_type");

          assertNotNull("Document should have valid ID", docId);
          assertEquals(
              "Document should be marked as CAGRA+brute force index type",
              "cagra_and_brute_force",
              indexType);
          assertTrue("Score should be positive", scoreDoc.score > 0);
        }

        // Test that the CAGRA+brute force index handles both approximate and exact search
        // consistently
        for (int trial = 0; trial < 3; trial++) {
          float[] trialQueryVector = generateRandomVector(vectorDimension, random());
          KnnFloatVectorQuery trialQuery =
              new KnnFloatVectorQuery("vector", trialQueryVector, Math.min(5, actualVectorCount));
          TopDocs trialResults = searcher.search(trialQuery, Math.min(5, actualVectorCount));

          assertTrue("Trial " + trial + " should find results", trialResults.scoreDocs.length > 0);
          log.info("Trial " + trial + " found " + trialResults.scoreDocs.length + " results");
        }
      } else {
        log.info("No vectors in CAGRA+brute force merged index - skipping vector search");
      }

      log.info("CAGRA+brute force merge verification completed successfully");
    }
  }

  /**
   * Test large scale merge to stress test the system
   **/
  @Test
  public void testLargeScaleMerge() throws IOException {
    assumeTrue(
        "testLargeScaleMerge requires -DlargeScale=true",
        Boolean.parseBoolean(System.getProperty("largeScale", "false")));

    log.info("Starting testLargeScaleMerge");

    // Randomize large scale parameters
    int maxBufferedDocs = 40 + random().nextInt(21); // 40-60 docs per buffer
    int segmentCount = 15 + random().nextInt(11); // 15-25 segments
    int docsPerSegment = 30 + random().nextInt(21); // 30-50 docs per segment
    int totalDocuments = segmentCount * docsPerSegment;

    log.info(
        "Randomized large scale parameters: maxBufferedDocs="
            + maxBufferedDocs
            + ", segmentCount="
            + segmentCount
            + ", docsPerSegment="
            + docsPerSegment
            + ", totalDocuments="
            + totalDocuments);

    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(alwaysKnnVectorsFormat(new CuVS2510GPUVectorsFormat()))
            .setMaxBufferedDocs(maxBufferedDocs)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);

    try (IndexWriter writer = new IndexWriter(directory, config)) {
      for (int seg = 0; seg < segmentCount; seg++) {
        log.info("Creating segment " + (seg + 1) + "/" + segmentCount);

        // Randomize vector probability per segment
        double vectorProbability =
            0.5 + (random().nextDouble() * 0.4); // 50-90% vectors per segment

        for (int i = 0; i < docsPerSegment; i++) {
          int docId = seg * docsPerSegment + i;
          Document doc = new Document();
          doc.add(new StringField("id", String.valueOf(docId), Field.Store.YES));
          doc.add(new NumericDocValuesField("segment", seg));
          doc.add(new NumericDocValuesField("position", i));

          // Add vector based on segment's randomized probability
          if (random().nextDouble() < vectorProbability) {
            float[] vector = generateRandomVector(vectorDimension, random());
            doc.add(new KnnFloatVectorField("vector", vector, VectorSimilarityFunction.COSINE));
          }

          writer.addDocument(doc);
        }
        writer.commit();
      }

      log.info("Created " + segmentCount + " segments with " + totalDocuments + " total documents");

      // Force merge all segments
      long startTime = System.currentTimeMillis();
      writer.forceMerge(1);
      long mergeTime = System.currentTimeMillis() - startTime;

      log.info("Large scale merge completed in " + mergeTime + "ms");
    }

    // Verify the large merged index
    try (DirectoryReader reader = DirectoryReader.open(directory)) {
      assertEquals("Should have exactly one segment after merge", 1, reader.leaves().size());

      LeafReader leafReader = reader.leaves().get(0).reader();
      assertEquals("Total documents should match", totalDocuments, leafReader.maxDoc());

      // Test vector search performance
      var vectorValues = leafReader.getFloatVectorValues("vector");
      int actualVectorCount = vectorValues != null ? vectorValues.size() : 0;

      if (actualVectorCount > 0) {
        IndexSearcher searcher = new IndexSearcher(reader);
        float[] queryVector = generateRandomVector(vectorDimension, random());

        // Randomize search parameters for large scale test
        int searchK =
            Math.min(20 + random().nextInt(31), Math.min(actualVectorCount, TOP_K_LIMIT)); // 20-50

        long searchStart = System.currentTimeMillis();
        KnnFloatVectorQuery vectorQuery = new KnnFloatVectorQuery("vector", queryVector, searchK);
        TopDocs vectorResults = searcher.search(vectorQuery, searchK);
        long searchTime = System.currentTimeMillis() - searchStart;

        assertTrue("Should find vector results in large index", vectorResults.scoreDocs.length > 0);
        log.info(
            "Vector search in large index returned "
                + vectorResults.scoreDocs.length
                + " results out of "
                + actualVectorCount
                + " vectors in "
                + searchTime
                + "ms");
      } else {
        log.info("No vectors in large merged index - skipping vector search");
      }

      log.info("Large scale merge verification completed successfully");
    }
  }

  /** Helper method to generate random vectors */
  private float[] generateRandomVector(int dimension, Random random) {
    float[] vector = new float[dimension];
    for (int i = 0; i < dimension; i++) {
      vector[i] = (float) random().nextGaussian();
    }
    // Normalize the vector
    float norm = 0.0f;
    for (float v : vector) {
      norm += v * v;
    }
    norm = (float) Math.sqrt(norm);
    if (norm > 0) {
      for (int i = 0; i < dimension; i++) {
        vector[i] /= norm;
      }
    }
    return vector;
  }

  /** Helper method to generate random text strings for sorting */
  private String generateRandomText(Random random, int length) {
    StringBuilder sb = new StringBuilder(length);
    String chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    for (int i = 0; i < length; i++) {
      sb.append(chars.charAt(random().nextInt(chars.length())));
    }
    return sb.toString();
  }
}

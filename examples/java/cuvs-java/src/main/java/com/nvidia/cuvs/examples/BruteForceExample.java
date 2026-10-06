/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.examples;

import com.nvidia.cuvs.BruteForceIndex;
import com.nvidia.cuvs.BruteForceIndexParams;
import com.nvidia.cuvs.BruteForceQuery;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.SearchResults;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.UUID;
import java.util.logging.Logger;

public class BruteForceExample {

  private static final Logger log = Logger.getLogger(BruteForceExample.class.getName());

  public static void main(String[] args) throws Throwable {

    // Sample data and query
    float[][] vectors = {
      {0.74021935f, 0.9209938f},
      {0.03902049f, 0.9689629f},
      {0.92514056f, 0.4463501f},
      {0.6673192f, 0.10993068f}
    };

    float[][] queries = {
      {0.48216683f, 0.0428398f},
      {0.5084142f, 0.6545497f},
      {0.51260436f, 0.2643005f},
      {0.05198065f, 0.5789965f}
    };

    try (CuVSResources resources = CuVSResources.create()) {

      // Create a query object with the query vectors
      BruteForceQuery cuvsQuery =
          new BruteForceQuery.Builder(resources).withTopK(3).withQueryVectors(queries).build();

      // Set index parameters
      BruteForceIndexParams indexParams = new BruteForceIndexParams.Builder().build();

      // Create the index with the dataset
      BruteForceIndex index =
          BruteForceIndex.newBuilder(resources)
              .withDataset(vectors)
              .withIndexParams(indexParams)
              .build();

      // Saving the index on to the disk.
      String indexFileName = UUID.randomUUID().toString() + ".bf";
      index.serialize(new FileOutputStream(indexFileName));

      // Loading a BRUTEFORCE index from disk.
      File indexFile = new File(indexFileName);
      InputStream inputStream = new FileInputStream(indexFile);
      BruteForceIndex loadedIndex = BruteForceIndex.newBuilder(resources).from(inputStream).build();

      // Perform the search
      SearchResults resultsFromLoadedIndex = loadedIndex.search(cuvsQuery);

      // Check results
      log.info(resultsFromLoadedIndex.getResults().toString());

      // Perform the search
      SearchResults results = index.search(cuvsQuery);

      // Check results
      log.info(results.getResults().toString());

      // Cleanup
      index.close();
      loadedIndex.close();

      if (indexFile.exists()) {
        indexFile.delete();
      }
    }
  }
}
